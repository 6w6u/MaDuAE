"""
MaDuAEImputer - Triple Tower Dual-Path Autoencoder Imputer
"""

import gc
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from kneed import KneeLocator
from torch.utils.data import random_split, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .models import GeneAutoencoder, TripleDualPathAE
from .dataset import IndexedTensorDataset
from .neighbors import compute_imputation_mask
from .clustering import consensus_leiden_clustering
from .training import train_triple_dual_path
from .imputation import impute_triple_dual_path


class MaDuAEImputer:
    """MaDuAE Imputer (Triple Tower Dual-Path Autoencoder Imputer)

    Args:
        hidden_dim (int): Autoencoder hidden dimension, default 32
        num_epochs (int): Maximum training epochs, default 300
        learning_rate (float): Learning rate, default 1e-3
        weight_decay (float): Weight decay, default 3e-4
        couple_weight (float): Coupling loss weight, default 1.0
        consist_weight (float or list): Consistency loss weight, default 0.1.
            Can be a single value (same for all towers) or a list of length 3 (different per tower)
        inter_tower_sim_weight (float): Inter-tower similarity consistency weight, default 0.1
        patience (int): Early stopping patience, default 15
        batch_size (int): Training batch size, default 1024
        train_ratio (float): Training set ratio, default 0.9
        n_neighbors_list (list): Neighbor count list for Leiden clustering, default [15, 20]
        resolution_list (list): Resolution list for Leiden clustering, default [0.3, 0.4, ..., 1.2]
        min_cells (int): Minimum cells for gene filtering, default 3
        n_pca_neighbors (int): Number of neighbors for PCA neighbor search, default 101, including self
        k_neighbors (int): Final number of neighbors to keep, default 11, including self
        device (str): Computing device, default 'cuda' (auto-detected)
        verbose (bool): Whether to print training logs, default True
        random_state (int): Random seed, default 0

    Example:
        >>> import scanpy as sc
        >>> from MaDuAEImpute import MaDuAEImputer
        >>>
        >>> # Impute from AnnData object, return AnnData
        >>> imputer = MaDuAEImputer(hidden_dim=32, num_epochs=300)
        >>> imputed_adata = imputer.fit_transform(adata, true_labels=labels)
        >>>
        >>> # Save best model weights
        >>> imputed_adata = imputer.fit_transform(adata, save_best_model="model.pth")

    Attributes:
        model_: Trained TripleDualPathAE model
        labels_: Dictionary of clustering labels used {'hvg': ..., 'pearson': ..., 'spearman': ...}
        mask_: Imputation mask
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_epochs: int = 300,
        learning_rate: float = 1e-3,
        weight_decay: float = 3e-4,
        couple_weight: float = 1.0,
        consist_weight= 0.1,
        inter_tower_sim_weight: float = 0.1,
        patience: int = 15,
        batch_size: int = 1024,
        train_ratio: float = 0.9,
        n_neighbors_list: list = None,
        resolution_list: list = None,
        min_cells: int = 3,
        n_pca_neighbors: int = 101,
        k_neighbors: int = 11,
        device: str = None,
        verbose: bool = True,
        random_state: int = 0,
    ):
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.couple_weight = couple_weight
        self.consist_weight = consist_weight
        self.inter_tower_sim_weight = inter_tower_sim_weight
        self.patience = patience
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        # 共识 Leiden 聚类参数
        if n_neighbors_list is None:
            n_neighbors_list = [15, 20]
        if resolution_list is None:
            resolution_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        self.n_neighbors_list = n_neighbors_list
        self.resolution_list = resolution_list
        self.min_cells = min_cells
        self.n_pca_neighbors = n_pca_neighbors
        self.k_neighbors = k_neighbors
        self.verbose = verbose
        self.random_state = random_state

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model_ = None
        self.labels_ = None
        self.mask_ = None
        self.ori_mask_ = None
        self._true_labels = None
        self._n_clusters = None

    def _evaluate_clustering(self, X_pca, labels_true, n_clusters):
        """Evaluate K-means clustering performance"""
        kmeans = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000,
                        random_state=self.random_state)
        kmeans.fit(X_pca)
        pred_labels = kmeans.labels_
        ari = adjusted_rand_score(labels_true, pred_labels)
        nmi = normalized_mutual_info_score(labels_true, pred_labels)
        return ari, nmi

    def _preprocess(self, adata: sc.AnnData):
        """Internal preprocessing: normalization, log1p, highly variable gene selection, PCA

        Consistent with original code:
        1. Normalization + log1p
        2. Save adataX_copy (full matrix after normalization+log1p)
        3. Highly variable gene selection
        4. Scale + PCA

        Returns:
            adata: Preprocessed AnnData (HVG subset, PCA computed)
            adataX_copy: Full expression matrix after normalization+log1p (all filtered genes)
            n_genes: Number of filtered genes (for model initialization)
            genes: Filtered gene names
        """
        # Filter lowly expressed genes
        sc.pp.filter_genes(adata, min_cells=self.min_cells)

        # Save filtered gene names (before adata is further modified)
        genes = adata.var.index.copy()

        # Normalization
        library_sizes = adata.X.sum(axis=1)
        median_lib_size = np.median(library_sizes)
        print(median_lib_size)
        sc.pp.normalize_total(adata, target_sum=max(1e4, median_lib_size))

        sc.pp.log1p(adata)

        # Save full matrix after normalization+log1p (all filtered genes)
        adataX_copy = adata.X.copy()
        n_genes = adataX_copy.shape[1]

        # Highly variable gene selection
        sc.pp.highly_variable_genes(adata)
        y = -np.sort(-adata.var['dispersions_norm'])
        x = np.arange(1, len(y) + 1)
        knee = KneeLocator(x, y, curve='convex', direction='decreasing',
                           interp_method='polynomial')
        if self.verbose:
            print(f'Number of highly variable genes: {knee.elbow}')

        mask_hvgs = adata.var['dispersions_norm'] > knee.elbow_y
        adata = adata[:, np.where(mask_hvgs)[0]]

        sc.pp.scale(adata)
        sc.pp.pca(adata, svd_solver='arpack')

        return adata, adataX_copy, n_genes, genes

    def _generate_three_labels(self, adataX_copy: np.ndarray, adata_proc: sc.AnnData):
        """Generate three types of clustering labels: HVG, Pearson, Spearman
        Args:
            adataX_copy: Preprocessed expression matrix
            adata_proc: Preprocessed AnnData object (PCA computed)

        Returns:
            labels_hvg, labels_pearson, labels_spearman: Three label arrays
        """
        from .clustering import (
            consensus_leiden_clustering,
            consensus_leiden_clustering_from_adata,
            prcomp_irlba_rotation,
            fast_spearman_corr
        )
        import time

        # 1. HVG labels - consensus Leiden clustering based on highly variable genes
        if self.verbose:
            print("Generating HVG labels...")
        start_time = time.time()
        labels_hvg = consensus_leiden_clustering(
            adata_proc,
            n_neighbors_list=self.n_neighbors_list,
            resolution_list=self.resolution_list
        )
        if self.verbose:
            print(f"HVG labels generated, elapsed time: {time.time() - start_time:.2f}s")

        # Calculate PCA dimension
        pca_dim = min(int(0.05 * adataX_copy.shape[0]), 100)
        if self.verbose:
            print(f"PCA dimension: {pca_dim}")

        # Create temporary AnnData for storing dimensionality reduction results
        import scanpy as sc
        adata_temp = sc.AnnData(X=adataX_copy)

        # 2. Pearson labels - consensus Leiden clustering after Pearson correlation dimensionality reduction
        if self.verbose:
            print("Generating Pearson labels...")
        start_time = time.time()
        corr_mat_pearson = np.corrcoef(adataX_copy)
        distance_matrix_pearson = 1 - corr_mat_pearson
        X_reduced_pearson = prcomp_irlba_rotation(
            distance_matrix_pearson, n_components=pca_dim, center=True, scale=True
        )
        adata_temp.obsm['X_pca'] = X_reduced_pearson
        labels_pearson = consensus_leiden_clustering_from_adata(
            adata_temp,
            n_neighbors_list=self.n_neighbors_list,
            resolution_list=self.resolution_list
        )
        if self.verbose:
            print(f"Pearson labels generated, elapsed time: {time.time() - start_time:.2f}s")

        # 3. Spearman labels - consensus Leiden clustering after Spearman correlation dimensionality reduction
        if self.verbose:
            print("Generating Spearman labels...")
        start_time = time.time()
        corr_mat_spearman = fast_spearman_corr(adataX_copy.T)
        distance_matrix_spearman = 1 - corr_mat_spearman
        X_reduced_spearman = prcomp_irlba_rotation(
            distance_matrix_spearman, n_components=pca_dim, center=True, scale=True
        )
        adata_temp.obsm['X_pca'] = X_reduced_spearman
        labels_spearman = consensus_leiden_clustering_from_adata(
            adata_temp,
            n_neighbors_list=self.n_neighbors_list,
            resolution_list=self.resolution_list
        )
        if self.verbose:
            print(f"Spearman labels generated, elapsed time: {time.time() - start_time:.2f}s")

        return labels_hvg, labels_pearson, labels_spearman

    def fit(self, adata: sc.AnnData,
            true_labels: np.ndarray = None, save_best_model: str = None):
        """Train the imputation model

        Args:
            adata: scanpy AnnData object (raw count matrix)
            true_labels: Optional, true label array (n_cells,), used to validate K-means clustering performance before and after imputation
            save_best_model: Optional, path to save model weights (.pth file)

        Returns:
            self
        """
        if self.verbose:
            print(f"Using device: {self.device}")

        # Save true labels and number of clusters
        if true_labels is not None:
            self._true_labels = np.array(true_labels)
            self._n_clusters = len(np.unique(true_labels))
            if self.verbose:
                print(f"Number of unique true labels: {self._n_clusters}")

        # Save original information
        cells = adata.obs.index
        n_cells = adata.shape[0]

        # Save raw count matrix for subsequent evaluation
        raw_counts = adata.X.copy()

        # Preprocessing
        adata_proc = adata.copy()
        adata_proc, adataX_copy, n_genes, genes = self._preprocess(adata_proc)

        # If true labels provided, evaluate performance before imputation
        if self._true_labels is not None:
            # Re-preprocess from raw counts for evaluation (using target_sum=10000)
            adata_eval_before = sc.AnnData(X=raw_counts)
            sc.pp.filter_genes(adata_eval_before, min_cells=self.min_cells)
            sc.pp.normalize_total(adata_eval_before, target_sum=1e4)
            sc.pp.log1p(adata_eval_before)
            sc.pp.highly_variable_genes(adata_eval_before)
            y = -np.sort(-adata_eval_before.var['dispersions_norm'])
            x = np.arange(1, len(y) + 1)
            knee = KneeLocator(x, y, curve='convex', direction='decreasing',
                               interp_method='polynomial')
            mask_hvgs = adata_eval_before.var['dispersions_norm'] > knee.elbow_y
            adata_eval_before = adata_eval_before[:, np.where(mask_hvgs)[0]]
            sc.pp.scale(adata_eval_before)
            sc.pp.pca(adata_eval_before, svd_solver='arpack')

            ari_before, nmi_before = self._evaluate_clustering(
                adata_eval_before.obsm['X_pca'], self._true_labels, self._n_clusters
            )
            if self.verbose:
                print(f"Before imputation K-means ARI: {ari_before:.4f}, NMI: {nmi_before:.4f}")
            self._ari_before = ari_before
            self._nmi_before = nmi_before

        # Generate three types of clustering labels (HVG, Pearson, Spearman)
        labels_hvg, labels_pearson, labels_spearman = self._generate_three_labels(
            adataX_copy, adata_proc
        )
        self.labels_ = {
            'hvg': labels_hvg,
            'pearson': labels_pearson,
            'spearman': labels_spearman
        }

        lbl_hvg = torch.tensor(labels_hvg, dtype=torch.long)
        lbl_pear = torch.tensor(labels_pearson, dtype=torch.long)
        lbl_spear = torch.tensor(labels_spearman, dtype=torch.long)
        lbl_list = [lbl_hvg, lbl_pear, lbl_spear]

        # Generate imputation mask
        if self.verbose:
            print("Computing imputation mask...")
        self.mask_, self.ori_mask_ = compute_imputation_mask(
            adataX_copy, adata_proc.obsm['X_pca'],
            n_pca_neighbors=self.n_pca_neighbors,
            k_neighbors=self.k_neighbors
        )
        positive_ratio = np.sum(self.mask_ > 0) / self.mask_.size
        if self.verbose:
            print(f'Mask positive ratio: {positive_ratio:.4f}')

        gc.collect()

        # Prepare data
        ori_data = torch.tensor(adataX_copy, dtype=torch.float32)
        mask_data = torch.tensor(1 - self.mask_, dtype=torch.float32)

        pseudo_labels_dummy = torch.zeros(ori_data.shape[0], dtype=torch.long)
        dataset_row = IndexedTensorDataset(ori_data, mask_data, pseudo_labels_dummy)
        n_total = len(dataset_row)
        train_size = int(self.train_ratio * n_total)
        valid_size = n_total - train_size
        train_ds_row, valid_ds_row = random_split(dataset_row, [train_size, valid_size])
        train_loader_row = DataLoader(train_ds_row, batch_size=self.batch_size, shuffle=True)
        valid_loader_row = DataLoader(valid_ds_row, batch_size=self.batch_size, shuffle=False)

        dataset_col = IndexedTensorDataset(
            ori_data.t(), mask_data.t(),
            torch.zeros(ori_data.shape[1], dtype=torch.long)
        )
        n_total_col = len(dataset_col)
        train_size_col = int(self.train_ratio * n_total_col)
        valid_size_col = n_total_col - train_size_col
        train_ds_col, valid_ds_col = random_split(dataset_col, [train_size_col, valid_size_col])
        train_loader_col = DataLoader(train_ds_col, batch_size=self.batch_size, shuffle=True)
        valid_loader_col = DataLoader(valid_ds_col, batch_size=self.batch_size, shuffle=False)

        # Model initialization - triple tower model
        model = TripleDualPathAE(
            n_cells, n_genes, hidden_dim=self.hidden_dim
        ).to(self.device)

        steps_per_epoch = int(train_size // self.batch_size) + 1
        if self.verbose:
            print(f"Steps per epoch: {steps_per_epoch}")

        # Training
        if self.verbose:
            print("Start training...")
        trained_model, _, _ = train_triple_dual_path(
            model=model,
            train_loader_row=train_loader_row,
            train_loader_col=train_loader_col,
            valid_loader_row=valid_loader_row,
            valid_loader_col=valid_loader_col,
            labels_list=lbl_list,
            steps_per_epoch=steps_per_epoch,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            couple_weight=self.couple_weight,
            consist_weight=self.consist_weight,
            inter_tower_sim_weight=self.inter_tower_sim_weight,
            patience=self.patience,
            device=self.device,
            verbose=self.verbose,
        )

        self.model_ = trained_model
        self._dataset_row = dataset_row
        self._dataset_col = dataset_col
        self._adataX_copy = adataX_copy
        self._cells = cells
        self._genes = genes
        self._adata_proc = adata_proc

        # Save model weights (if specified)
        if save_best_model is not None:
            self.save_model(save_best_model)

        return self

    def transform(self, return_adata: bool = True) -> sc.AnnData:
        """Perform imputation

        Args:
            return_adata: Whether to return AnnData object, default True

        Returns:
            If return_adata=True: AnnData object with imputation results saved in .X
            If return_adata=False: (n_cells, n_genes) imputed expression matrix
        """
        if self.model_ is None:
            raise RuntimeError("Model not trained yet, please call fit() first.")

        X_ens = impute_triple_dual_path(
            self.model_, self._dataset_row, self._dataset_col,
            device=self.device, batch_size=256, #self.batch_size
            alpha=0.5
        )
        imp_result = np.where(self.ori_mask_ == 1, X_ens, self._adataX_copy)

        if self.verbose:
            print(f"Imputation completed! Result shape: {imp_result.shape}")

        if not return_adata:
            return imp_result

        # Build returned AnnData object
        adata_imputed = sc.AnnData(
            X=imp_result,
            obs=pd.DataFrame(index=self._cells),
            var=pd.DataFrame(index=self._genes)
        )

        # If true labels provided, evaluate performance after imputation
        if self._true_labels is not None:
            # Create AnnData copy for evaluation (using target_sum=10000)
            adata_eval = adata_imputed.copy()
            adata_eval.X = np.expm1(adata_eval.X)  # Recover from log1p
            sc.pp.normalize_total(adata_eval, target_sum=1e4)
            sc.pp.log1p(adata_eval)
            sc.pp.highly_variable_genes(adata_eval)
            y = -np.sort(-adata_eval.var['dispersions_norm'])
            x = np.arange(1, len(y) + 1)
            knee = KneeLocator(x, y, curve='convex', direction='decreasing',
                               interp_method='polynomial')
            mask_hvgs = adata_eval.var['dispersions_norm'] > knee.elbow_y
            adata_eval = adata_eval[:, np.where(mask_hvgs)[0]]
            sc.pp.scale(adata_eval)
            sc.pp.pca(adata_eval, svd_solver='arpack')

            ari_after, nmi_after = self._evaluate_clustering(
                adata_eval.obsm['X_pca'], self._true_labels, self._n_clusters
            )
            if self.verbose:
                print(f"After imputation K-means ARI: {ari_after:.4f}, NMI: {nmi_after:.4f}")
                print(f"ARI improvement: {ari_after - self._ari_before:.4f}")
                print(f"NMI improvement: {nmi_after - self._nmi_before:.4f}")

            # Save evaluation results to AnnData uns
            adata_imputed.uns['clustering_metrics'] = {
                'ari_before': self._ari_before,
                'nmi_before': self._nmi_before,
                'ari_after': ari_after,
                'nmi_after': nmi_after,
                'ari_improvement': ari_after - self._ari_before,
                'nmi_improvement': nmi_after - self._nmi_before,
            }

        return adata_imputed

    def fit_transform(self, adata: sc.AnnData,
                      true_labels: np.ndarray = None, save_best_model: str = None) -> sc.AnnData:
        """Train model and perform imputation (one step)

        Args:
            adata: scanpy AnnData object (raw count matrix)
            true_labels: Optional, true label array, used to validate K-means clustering performance before and after imputation
            save_best_model: Optional, path to save model weights (.pth file)

        Returns:
            AnnData object with imputation results saved in .X
        """
        self.fit(adata, true_labels=true_labels, save_best_model=save_best_model)
        return self.transform(return_adata=True)

    def save_model(self, path: str):
        """Save model weights

        Args:
            path: Save path (.pth file)
        """
        if self.model_ is None:
            raise RuntimeError("Model not trained yet, cannot save.")
        torch.save(self.model_.state_dict(), path)
        if self.verbose:
            print(f"Model saved to: {path}")

    def load_model(self, path: str, n_cells: int, n_genes: int):
        """Load model weights

        Args:
            path: Model weights path (.pth file)
            n_cells: Number of cells
            n_genes: Number of genes
        """
        model = TripleDualPathAE(
            n_cells, n_genes, hidden_dim=self.hidden_dim
        ).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        self.model_ = model
        if self.verbose:
            print(f"Model loaded from: {path}")
