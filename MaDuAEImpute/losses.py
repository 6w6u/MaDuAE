"""
Loss function definitions

Contains:
- supervised_contrastive_clustering_loss: Supervised contrastive learning clustering consistency loss
"""

import torch
import torch.nn.functional as F


def supervised_contrastive_clustering_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                                            temperature: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Learning Clustering Consistency Loss

    Based on the Supervised Contrastive Learning (SupCon) framework, by pulling together
    latent space representations of samples within the same cluster and pushing apart
    representations of samples from different clusters, enhancing the distinguishability
    of cluster structures in the latent space.

    Algorithm steps:
    1. L2 normalize latent space encodings, mapping representations to unit hypersphere
    2. Compute cosine similarity matrix between samples
    3. Build positive sample mask based on labels (same-class sample pairs)
    4. Optimize using InfoNCE loss to bring positive sample pairs closer in latent space

    Mathematical form:
        L = -log[ exp(sim_ij / τ) / Σ_k exp(sim_ik / τ) ]
    where sim_ij is the cosine similarity between samples i and j, and τ is the temperature parameter.

    Args:
        embeddings: (N, d) Latent space encodings, N is number of samples, d is encoding dimension
        labels: (N,) Cluster labels (integers), cluster category each sample belongs to
        temperature: Temperature parameter τ, controls sharpness of similarity distribution.
                     Smaller values make distribution sharper, enhancing distinguishability;
                     Larger values make distribution smoother, increasing stability.
                     Default 0.1

    Returns:
        Scalar loss value, representing average contrastive loss for current batch of samples

    References:
        Khosla et al. "Supervised Contrastive Learning". NeurIPS 2020.
    """
    device = embeddings.device
    N = embeddings.size(0)
    if N < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # L2 归一化 → 余弦相似度 = 点积
    embeddings = F.normalize(embeddings, dim=1)
    # 计算相似度矩阵 (N, N)
    sim = torch.matmul(embeddings, embeddings.t()) / temperature
    # 构建正样本掩码（同类且非自身）
    labels = labels.unsqueeze(1)
    mask_pos = (labels == labels.t()).float()
    mask_pos.fill_diagonal_(0)

    if mask_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 数值稳定技巧
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()
    exp_sim = torch.exp(sim_stable)
    exp_sum = exp_sim.sum(dim=1, keepdim=True)

    log_prob = sim_stable - torch.log(exp_sum + 1e-8)
    loss_per_anchor = -(mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    return loss_per_anchor.mean()
