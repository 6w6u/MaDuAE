"""
MaDuAEImpute - Triple Tower Dual-Path Autoencoder Imputation Algorithm

A scRNA-seq data imputation method based on triple dual-path autoencoder architecture
that jointly models cell and gene representations with cross-path consistency.
"""

__version__ = "1.0.0"

from .maDuAE import MaDuAEImputer
from .models import Autoencoder, GeneAutoencoder, DualPathTower, TripleDualPathAE
from .imputation import impute_triple_dual_path
from .training import train_triple_dual_path
from .losses import supervised_contrastive_clustering_loss

__all__ = [
    "MaDuAEImputer",
    "Autoencoder",
    "GeneAutoencoder",
    "DualPathTower",
    "TripleDualPathAE",
    "impute_triple_dual_path",
    "train_triple_dual_path",
    "supervised_contrastive_clustering_loss",
]
