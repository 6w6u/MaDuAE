"""
Clustering helper functions

Contains:
- recursive_leiden: Leiden clustering
- consensus_leiden_clustering: Multi-resolution consensus Leiden clustering
- consensus_leiden_clustering_from_adata: Consensus Leiden clustering from AnnData's X_pca
- prcomp_irlba_rotation: PCA dimensionality reduction using truncated SVD (reproducing R's prcomp_irlba)
- fast_spearman_corr: Fast computation of Spearman correlation coefficient
"""

import numpy as np
import scanpy as sc
import igraph as ig
import leidenalg
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata


def recursive_leiden(adata, resolution: float = 1.0) -> np.ndarray:
    """Perform Leiden clustering

    Args:
        adata: scanpy AnnData object (neighbors must be computed)
        resolution: Resolution parameter

    Returns:
        Clustering label array
    """
    sc.tl.leiden(adata, resolution=resolution, random_state=0)
    best_clusters = adata.obs['leiden'].astype(int)
    return best_clusters


def _consensus_from_clusters(all_clusters: np.ndarray) -> np.ndarray:
    """Compute consensus labels from multiple clustering results"""
    num_objects = len(all_clusters[0])

    # Build co-occurrence matrix (float32 to avoid int64 memory blowup)
    similarity_matrix = np.zeros((num_objects, num_objects), dtype=np.float32)
    for labels in all_clusters:
        labels = np.array(labels)
        bool_matrix = labels[:, None] == labels
        similarity_matrix += bool_matrix
        del bool_matrix

    np.fill_diagonal(similarity_matrix, 0)
    # Normalize in-place to avoid creating an extra copy
    similarity_matrix /= len(all_clusters)

    # Build weighted graph and run Leiden
    coo = sparse.coo_matrix(similarity_matrix)
    del similarity_matrix
    coo_ut = sparse.triu(coo, k=1, format="coo")
    rows = coo_ut.row.astype(np.int32, copy=False)
    cols = coo_ut.col.astype(np.int32, copy=False)
    weights = coo_ut.data.astype(np.float32, copy=False)

    del coo, coo_ut

    G = ig.Graph(n=num_objects, edges=zip(rows, cols), directed=False)
    G.es["weight"] = weights

    del rows, cols, weights

    partition = leidenalg.find_partition(G, leidenalg.SurpriseVertexPartition, weights='weight')
    labels = np.array(partition.membership)

    return labels


def consensus_leiden_clustering(adata, n_neighbors_list=None, resolution_list=None,
                                 n_pcs: int = 50) -> np.ndarray:
    """Multi-resolution consensus Leiden clustering

    Run Leiden clustering at multiple neighbor counts and resolutions,
    aggregate into final labels through consensus graph.

    Args:
        adata: scanpy AnnData object (PCA must be computed)
        n_neighbors_list: List of neighbor counts, default [15, 20]
        resolution_list: List of resolutions, default [0.3, 0.4, ..., 1.2]
        n_pcs: Number of PCA principal components to use

    Returns:
        labels: Consensus clustering label array
    """
    if n_neighbors_list is None:
        n_neighbors_list = [15, 20]
    if resolution_list is None:
        resolution_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    all_clusters = []
    for nei in n_neighbors_list:
        sc.pp.neighbors(adata, n_neighbors=nei, metric='cosine', n_pcs=n_pcs)
        for res in resolution_list:
            labels = recursive_leiden(adata, resolution=res)
            all_clusters.append(labels)

    all_clusters = np.array(all_clusters)
    return _consensus_from_clusters(all_clusters)


def consensus_leiden_clustering_from_adata(adata, n_neighbors_list=None,
                                           resolution_list=None) -> np.ndarray:
    """Multi-resolution consensus Leiden clustering from AnnData's X_pca

    Args:
        adata: scanpy AnnData object (X_pca must be computed)
        n_neighbors_list: List of neighbor counts, default [15, 20]
        resolution_list: List of resolutions, default [0.3, 0.4, ..., 1.2]

    Returns:
        labels: Consensus clustering label array
    """
    if n_neighbors_list is None:
        n_neighbors_list = [15, 20]
    if resolution_list is None:
        resolution_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    n_pcs = adata.obsm['X_pca'].shape[1]

    all_clusters = []
    for nei in n_neighbors_list:
        sc.pp.neighbors(adata, n_neighbors=nei, metric='cosine', n_pcs=n_pcs)
        for res in resolution_list:
            labels = recursive_leiden(adata, resolution=res)
            all_clusters.append(labels)

    all_clusters = np.array(all_clusters)
    return _consensus_from_clusters(all_clusters)


def prcomp_irlba_rotation(X, n_components, center=True, scale=True):
    """Reproduce R's prcomp_irlba(X, n = n_components, center=TRUE, scale.=TRUE)$rotation

    Parameters:
        X: Data matrix (n_samples, n_features) — processed normally even for correlation matrices
        n_components: Number of principal components (k)
        center: Whether to center
        scale: Whether to scale

    Returns:
        rotation: Loadings matrix (n_features, n_components)
    """
    X = np.array(X)

    # Step 1: Preprocessing (center and scale)
    if center or scale:
        if X.shape[1] < 10000:
            scaler = StandardScaler(with_mean=center, with_std=scale)
            X_processed = scaler.fit_transform(X)  # Transform each column of input matrix
        else:
            scaler = StandardScaler(with_mean=center, with_std=scale)

            # Assuming you have a generator or can read data in chunks
            chunk_size = 10000  # Adjust based on memory

            # Step 1: Fit scaler in chunks (compute global mean/std)
            for i in range(0, len(X), chunk_size):
                chunk = X[i:i + chunk_size]
                scaler.partial_fit(chunk)

            # Step 2: Transform in chunks
            X_processed_chunks = []
            for i in range(0, len(X), chunk_size):
                chunk = X[i:i + chunk_size]
                chunk_scaled = scaler.transform(chunk)
                X_processed_chunks.append(chunk_scaled)

            # Finally concatenate
            X_processed = np.vstack(X_processed_chunks)
    else:
        X_processed = X

    # Step 2: Use svds to compute first n_components singular vectors
    k = n_components
    # Perform truncated SVD
    U, s, Vt = svds(X_processed, k=k)
    # Sort by singular values from largest to smallest
    idx = np.argsort(s)[::-1]
    Vt = Vt[idx, :]
    # Take first n_components right singular vectors (if k < n_components, will be fewer; but usually k=n_components)
    Vt = Vt[:n_components]
    # rotation = transpose of right singular vectors (columns are principal component directions)
    rotation = Vt.T  # shape: (n_features, n_components)

    return rotation


def fast_spearman_corr(X):
    """Fast computation of Spearman correlation coefficient

    Args:
        X: Data matrix (n_samples, n_features)

    Returns:
        corr_mat: Spearman correlation coefficient matrix (n_features, n_features)
    """
    X_rank = rankdata(X, axis=0, method='average')
    return np.corrcoef(X_rank, rowvar=False)
