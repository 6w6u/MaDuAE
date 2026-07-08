"""
Imputation logic

Contains:
- impute_single_dual_path: Single tower dual-path model imputation function
- impute_triple_dual_path: Triple tower dual-path model imputation function
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def impute_single_dual_path(model, dataset_row, dataset_col,
                            device: str = 'cuda', batch_size: int = 1024,
                            alpha: float = 0.5) -> np.ndarray:
    """Perform imputation using trained dual-path model

    Generate imputation matrices through Cell path and Gene path separately, then fuse with weights.

    Args:
        model: Trained DualPathTower model
        dataset_row: Cell path IndexedTensorDataset
        dataset_col: Gene path IndexedTensorDataset
        device: Computing device
        batch_size: Inference batch size
        alpha: Cell path weight (Gene path weight is 1-alpha)

    Returns:
        X_ens: (n_cells, n_genes) Fused imputation matrix
    """
    model.eval()
    n_cells = len(dataset_row)
    n_genes = len(dataset_col)

    # Single output matrix, accumulated in-place to avoid holding X_cell and X_gene simultaneously
    X_ens = np.zeros((n_cells, n_genes), dtype=np.float32)

    # Cell path - accumulate directly into X_ens
    loader_row = DataLoader(dataset_row, batch_size=batch_size, shuffle=False)
    for batch_idx, (data_row, _, _, _) in enumerate(loader_row):
        data_row = data_row.to(device)
        start = batch_idx * batch_size
        end = min(start + batch_size, n_cells)
        with torch.no_grad():
            avg_recon, _ = model.cell_ae(data_row)
            X_ens[start:end] = alpha * avg_recon.detach().cpu().numpy()
        del data_row, avg_recon

    # Gene path - accumulate directly into X_ens
    loader_col = DataLoader(dataset_col, batch_size=batch_size, shuffle=False)
    for _, (data_col, _, _, col_idx) in enumerate(loader_col):
        data_col = data_col.to(device)
        col_idx_np = col_idx.cpu().numpy()
        with torch.no_grad():
            avg_recon_T, _ = model.gene_ae(data_col)
            X_ens[:, col_idx_np] += (1 - alpha) * avg_recon_T.detach().cpu().T.numpy()
        del data_col, avg_recon_T, col_idx, col_idx_np

    np.clip(X_ens, 0, None, out=X_ens)
    return X_ens


def impute_triple_dual_path(model, dataset_row, dataset_col,
                            device: str = 'cuda', batch_size: int = 1024,
                            alpha: float = 0.5) -> np.ndarray:
    """Perform imputation using trained triple tower dual-path model

    Generate imputation matrices through three Cell paths and shared Gene path separately,
    then average and fuse with weights.

    Args:
        model: Trained TripleDualPathAE model
        dataset_row: Cell path IndexedTensorDataset
        dataset_col: Gene path IndexedTensorDataset
        device: Computing device
        batch_size: Inference batch size
        alpha: Cell path weight (Gene path weight is 1-alpha)

    Returns:
        X_ens: (n_cells, n_genes) Fused imputation matrix
    """
    model.eval()
    n_cells = len(dataset_row)
    n_genes = len(dataset_col)

    # Single output matrix, accumulated in-place to avoid holding X_cell and X_gene simultaneously
    X_ens = np.zeros((n_cells, n_genes), dtype=np.float32)

    # Cell path (average of three towers) - accumulate directly into X_ens
    loader_row = DataLoader(dataset_row, batch_size=batch_size, shuffle=False)
    for batch_idx, (data_row, _, _, _) in enumerate(loader_row):
        data_row = data_row.to(device)
        start = batch_idx * batch_size
        end = min(start + batch_size, n_cells)
        with torch.no_grad():
            recons = [tower.cell_ae(data_row)[0] for tower in model.towers]
            # Average three towers without torch.stack to save memory
            avg_recon = sum(recons) / 3.0
            X_ens[start:end] = alpha * avg_recon.detach().cpu().numpy()
        del data_row, avg_recon, recons

    # Gene path (three towers share the same GeneAE) - accumulate directly into X_ens
    loader_col = DataLoader(dataset_col, batch_size=batch_size, shuffle=False)
    for _, (data_col, _, _, col_idx) in enumerate(loader_col):
        data_col = data_col.to(device)
        col_idx_np = col_idx.cpu().numpy()
        with torch.no_grad():
            recons_T = [tower.gene_ae(data_col)[0] for tower in model.towers]
            # Average three towers without torch.stack to save memory
            avg_recon_T = sum(recons_T) / 3.0
            X_ens[:, col_idx_np] += (1 - alpha) * avg_recon_T.detach().cpu().T.numpy()
        del data_col, avg_recon_T, recons_T, col_idx, col_idx_np

    np.clip(X_ens, 0, None, out=X_ens)
    return X_ens
