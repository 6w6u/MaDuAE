"""
Training logic

Contains:
- train_single_dual_path: Single tower dual-path model training function
- train_triple_dual_path: Triple tower dual-path model training function
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .losses import supervised_contrastive_clustering_loss

def train_triple_dual_path(
        model,
        train_loader_row,
        train_loader_col,
        valid_loader_row,
        valid_loader_col,
        labels_list,
        num_epochs: int = 300,
        learning_rate: float = 1e-3,
        weight_decay: float = 3e-4,
        couple_weight: float = 1.0,
        consist_weight=0.1,
        inter_tower_sim_weight: float = 0.1,
        patience: int = 15,
        steps_per_epoch: int = None,
        device: str = 'cuda',
        verbose: bool = True,
):
    """Train triple tower dual-path model

    Args:
        model: TripleDualPathAE model instance
        train_loader_row: Cell path training DataLoader
        train_loader_col: Gene path training DataLoader
        valid_loader_row: Cell path validation DataLoader
        valid_loader_col: Gene path validation DataLoader
        labels_list: List[Tensor], 3 supervised clustering label Tensors, each (n_cells,)
        num_epochs: Maximum training epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay
        couple_weight: Coupling loss weight
        consist_weight: Consistency loss weight (can be a single value or list of length 3)
        inter_tower_sim_weight: Inter-tower similarity consistency weight
        patience: Early stopping patience
        steps_per_epoch: Steps per epoch, defaults to number of training batches
        device: Computing device
        verbose: Whether to print training logs

    Returns:
        model: Trained model
        best_combined_loss: Best combined loss
        best_valid_total: Best validation total loss
    """
    if isinstance(consist_weight, (int, float)):
        consist_weight = [consist_weight] * 3

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_combined_loss = float('inf')
    best_valid_total = float('inf')
    best_model_wts = None
    no_improve = 0

    labels_list = [lbl.to(device) for lbl in labels_list]

    for epoch in range(num_epochs):
        model.train()
        total_recon_row = total_recon_col = total_couple = total_consist = total_inter_tower_sim = 0.0
        n_batches = 0

        row_iter = iter(train_loader_row)
        col_iter = iter(train_loader_col)
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader_row)

        for _ in range(steps_per_epoch):
            try:
                data_row, weight_row, _, row_idx = next(row_iter)
            except StopIteration:
                row_iter = iter(train_loader_row)
                data_row, weight_row, _, row_idx = next(row_iter)
            try:
                data_col, weight_col, _, col_idx = next(col_iter)
            except StopIteration:
                col_iter = iter(train_loader_col)
                data_col, weight_col, _, col_idx = next(col_iter)

            data_row, weight_row = data_row.to(device), weight_row.to(device)
            data_col, weight_col = data_col.to(device), weight_col.to(device)
            row_idx, col_idx = row_idx.to(device), col_idx.to(device)

            optimizer.zero_grad()
            tower_outputs = model(data_row, data_col, row_idx, col_idx)

            loss_total = 0.0
            z_rows = []
            for i, (recon_row, recon_col, z_row, z_col, couple_loss) in enumerate(tower_outputs):
                loss_r = torch.sum(weight_row * (recon_row - data_row) ** 2) / (weight_row.sum() + 1e-8)
                loss_c = torch.sum(weight_col * (recon_col - data_col) ** 2) / (weight_col.sum() + 1e-8)

                labels_i_batch = labels_list[i][row_idx]
                consist_loss = supervised_contrastive_clustering_loss(z_row, labels_i_batch, temperature=0.1)
                tower_loss = loss_r + loss_c + couple_weight * couple_loss + consist_weight[i] * consist_loss
                loss_total += tower_loss
                z_rows.append(z_row.detach())

                total_recon_row += loss_r.item()
                total_recon_col += loss_c.item()
                total_couple += couple_loss.item()
                total_consist += consist_loss.item()

            # 三塔相似性一致性损失
            inter_tower_sim_loss = torch.tensor(0.0, device=device)
            if inter_tower_sim_weight > 0 and len(z_rows) == 3:
                sim_mats = []
                for z in z_rows:
                    z_norm = nn.functional.normalize(z, p=2, dim=1)
                    sim = torch.mm(z_norm, z_norm.t())
                    sim_mats.append(sim)
                inter_tower_sim_loss = (nn.functional.mse_loss(sim_mats[0], sim_mats[1]) +
                                        nn.functional.mse_loss(sim_mats[0], sim_mats[2]) +
                                        nn.functional.mse_loss(sim_mats[1], sim_mats[2])) / 3.0
                total_inter_tower_sim += inter_tower_sim_loss.item()

            loss_total = loss_total / 3.0 + inter_tower_sim_weight * inter_tower_sim_loss
            loss_total.backward()
            optimizer.step()
            n_batches += 1

        # 验证阶段
        model.eval()
        valid_recon_row = valid_recon_col = 0.0
        n_valid_row = len(valid_loader_row)
        n_valid_col = len(valid_loader_col)
        with torch.no_grad():
            for data_row, weight_row, _, _ in valid_loader_row:
                data_row, weight_row = data_row.to(device), weight_row.to(device)
                recons = [tower.cell_ae(data_row)[0] for tower in model.towers]
                avg_recon = torch.stack(recons).mean(dim=0)
                loss = torch.sum(weight_row * (avg_recon - data_row) ** 2) / (weight_row.sum() + 1e-8)
                valid_recon_row += loss.item()
            for data_col, weight_col, _, _ in valid_loader_col:
                data_col, weight_col = data_col.to(device), weight_col.to(device)
                recons_T = [tower.gene_ae(data_col)[0] for tower in model.towers]
                avg_recon_T = torch.stack(recons_T).mean(dim=0)
                loss = torch.sum(weight_col * (avg_recon_T - data_col) ** 2) / (weight_col.sum() + 1e-8)
                valid_recon_col += loss.item()

        avg_valid_row = valid_recon_row / n_valid_row if n_valid_row else 0
        avg_valid_col = valid_recon_col / n_valid_col if n_valid_col else 0
        valid_total = avg_valid_row + avg_valid_col

        avg_recon_row = total_recon_row / (3 * n_batches)
        avg_recon_col = total_recon_col / (3 * n_batches)
        avg_couple = total_couple / (3 * n_batches)
        avg_consist = total_consist / (3 * n_batches)
        avg_inter_tower_sim = total_inter_tower_sim / n_batches

        combined = (avg_recon_row + avg_recon_col) + (avg_valid_row + avg_valid_col)
        if combined < best_combined_loss:
            best_combined_loss = combined
            best_model_wts = copy.deepcopy(model.state_dict())
        if valid_total < best_valid_total:
            best_valid_total = valid_total
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step(valid_total)

        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"[Epoch {epoch + 1:3d}] "
                  f"Train: R_row={avg_recon_row:.4f}, R_col={avg_recon_col:.4f}, "
                  f"couple={avg_couple:.4f}, consist={avg_consist:.4f}, "
                  f"inter_sim={avg_inter_tower_sim:.6f} | "
                  f"Valid: row={avg_valid_row:.4f}, col={avg_valid_col:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"-> Early stop at epoch {epoch + 1}")
            break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model, best_combined_loss, best_valid_total
