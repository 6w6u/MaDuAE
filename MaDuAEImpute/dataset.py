"""
Dataset definitions

Contains:
- IndexedTensorDataset: PyTorch Dataset that supports returning indices
"""

from torch.utils.data import Dataset
import torch


class IndexedTensorDataset(Dataset):
    """Tensor dataset with indices

    Used to track the original index of each sample during training to support
    the calculation of coupling loss and consistency loss.

    Args:
        data (torch.Tensor): Expression matrix (N, G) or its transpose
        weight (torch.Tensor): Weight/mask matrix, same shape as data
        labels (torch.Tensor): Label vector (N,)
    """

    def __init__(self, data: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.weight = weight
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.weight[idx], self.labels[idx], idx
