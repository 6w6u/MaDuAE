"""
Neural network model definitions

Contains:
- Autoencoder: Cell autoencoder
- GeneAutoencoder: Gene autoencoder
- DualPathTower: Single tower dual-path model (CellAE + GeneAE)
- TripleDualPathAE: Triple tower dual-path model (3 CellAEs + shared GeneAE)
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Cell Autoencoder

    Encodes high-dimensional gene expression vectors of cells to low-dimensional latent space, then decodes to reconstruct.

    Args:
        input_dim (int): Input dimension (number of genes)
        hidden_dim (int): Hidden space dimension
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),
            nn.Linear(128, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.PReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, input_dim) Input expression matrix

        Returns:
            decoded: (batch_size, input_dim) Reconstructed expression matrix
            encoded: (batch_size, hidden_dim) Latent space encoding
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class GeneAutoencoder(nn.Module):
    """Gene Autoencoder

    Encodes high-dimensional cell expression vectors of genes to low-dimensional latent space, then decodes to reconstruct.
    Same structure as Autoencoder, but semantically used for gene dimension.

    Args:
        input_dim (int): Input dimension (number of cells)
        hidden_dim (int): Hidden space dimension
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),
            nn.Linear(128, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.PReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, input_dim) Transposed rows of input expression matrix

        Returns:
            decoded: (batch_size, input_dim) Reconstructed expression matrix
            encoded: (batch_size, hidden_dim) Latent space encoding
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class DualPathTower(nn.Module):
    """Single Tower Dual-Path Model

    Dual-path architecture integrating CellAE (independent) and GeneAE (externally passed),
    achieving cell-gene joint modeling through cross-region consistency loss.

    Args:
        n_cells (int): Number of cells
        n_genes (int): Number of genes
        hidden_dim (int): Hidden space dimension, default 32
        shared_gene_ae (GeneAutoencoder): Shared gene autoencoder instance
    """

    def __init__(self, n_cells: int, n_genes: int, hidden_dim: int = 32,
                 shared_gene_ae: GeneAutoencoder = None):
        super().__init__()
        self.cell_ae = Autoencoder(input_dim=n_genes, hidden_dim=hidden_dim)
        self.gene_ae = shared_gene_ae  # 必须非 None

    def forward(self, x_row: torch.Tensor, x_col: torch.Tensor,
                row_idx: torch.Tensor = None, col_idx: torch.Tensor = None):
        """
        Args:
            x_row: (B_row, G) Batch cell expression
            x_col: (B_col, C) Batch gene expression (i.e., rows of X^T)
            row_idx: (B_row,) Optional, for coupling loss
            col_idx: (B_col,) Optional, for coupling loss

        Returns:
            recon_row: (B_row, G) Cell path reconstruction
            recon_col: (B_col, C) Gene path reconstruction
            z_row: (B_row, d) Cell latent encoding
            z_col: (B_col, d) Gene latent encoding
            couple_loss: Cross-region consistency loss
        """
        recon_row, z_row = self.cell_ae(x_row)  # (B_row, G), (B_row, d)
        recon_col, z_col = self.gene_ae(x_col)  # (B_col, C), (B_col, d)

        couple_loss = torch.tensor(0.0, device=x_row.device)
        if row_idx is not None and col_idx is not None:
            # Cross-region consistency: recon_row[:, col_idx] vs recon_col[:, row_idx].T
            cross_from_row = recon_row[:, col_idx]  # (B_row, B_col)
            cross_from_col = recon_col[:, row_idx].t()  # (B_row, B_col)
            couple_loss = torch.mean((cross_from_row - cross_from_col) ** 2)

        return recon_row, recon_col, z_row, z_col, couple_loss


class TripleDualPathAE(nn.Module):
    """Triple Tower Dual-Path Autoencoder

    Contains 3 DualPathTowers, each tower has an independent CellAE, but shares the same GeneAE.
    Supports similarity consistency constraints between towers.

    Args:
        n_cells (int): Number of cells
        n_genes (int): Number of genes
        hidden_dim (int): Hidden space dimension, default 32
    """

    def __init__(self, n_cells: int, n_genes: int, hidden_dim: int = 32):
        super().__init__()
        # Create unique shared gene autoencoder
        self.shared_gene_ae = GeneAutoencoder(input_dim=n_cells, hidden_dim=hidden_dim)
        # Create three towers, each tower has independent CellAE, GeneAE points to same instance
        self.towers = nn.ModuleList([
            DualPathTower(n_cells, n_genes, hidden_dim, shared_gene_ae=self.shared_gene_ae)
            for _ in range(3)
        ])

    def forward(self, x_row: torch.Tensor, x_col: torch.Tensor,
                row_idx: torch.Tensor = None, col_idx: torch.Tensor = None):
        """
        Args:
            x_row: (B_row, G) Batch cell expression
            x_col: (B_col, C) Batch gene expression (i.e., rows of X^T)
            row_idx: (B_row,) Optional, for coupling loss
            col_idx: (B_col,) Optional, for coupling loss

        Returns:
            outputs: List[Tuple], each tower returns (recon_row, recon_col, z_row, z_col, couple_loss)
        """
        outputs = []
        for tower in self.towers:
            out = tower(x_row, x_col, row_idx, col_idx)
            outputs.append(out)
        return outputs
