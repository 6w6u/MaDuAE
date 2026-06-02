"""
Neighbor computation and mask generation

Contains:
- get_final_neighbors: Fuse cosine and Euclidean neighbors
- build_adjacency_matrix_fast: Build adjacency matrix
- find_neighbors: Extract neighbor list from adjacency matrix
- calculate_expression_rates: Calculate neighbor expression rates
- keep_top_half_nonzero_per_row_vectorized: Keep top 50% non-zero values per row
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_final_neighbors(cosine_indices: np.ndarray, euclidean_indices: np.ndarray,
                        k: int = 10):
    """Fuse cosine and Euclidean distance neighbor lists

    Take common neighbors of both distance metrics, sort by average rank, and take top-k.

    Args:
        cosine_indices: (N, K_cos) Cosine nearest neighbor indices
        euclidean_indices: (N, K_euc) Euclidean nearest neighbor indices
        k: Number of neighbors to keep per cell

    Returns:
        final_neighbors: Neighbor list
        diff_points: Differential points list (empty list in current implementation)
    """
    n_cells = cosine_indices.shape[0]
    final_neighbors = []
    for i in range(n_cells):
        cos_neighbors = set(cosine_indices[i])
        euc_neighbors = set(euclidean_indices[i])
        common_neighbors = cos_neighbors.intersection(euc_neighbors)
        if not common_neighbors:
            continue
        neighbor_ranks = []
        for neighbor in common_neighbors:
            cos_rank = np.where(cosine_indices[i] == neighbor)[0][0]
            euc_rank = np.where(euclidean_indices[i] == neighbor)[0][0]
            avg_rank = (cos_rank + euc_rank) / 2
            neighbor_ranks.append((neighbor, avg_rank))
        sorted_neighbors = sorted(neighbor_ranks, key=lambda x: x[1])
        top_k = sorted_neighbors[:k] if len(sorted_neighbors) > k else sorted_neighbors
        final_neighbors.append([n[0] for n in top_k])
    return final_neighbors, []


def build_adjacency_matrix_fast(final_indices: list) -> np.ndarray:
    """Build symmetric adjacency matrix from neighbor list

    Args:
        final_indices: Neighbor list, each element is a list of neighbor indices for that cell

    Returns:
        adjacency_matrix: (N, N) Symmetric adjacency matrix
    """
    n_cells = len(final_indices)
    adjacency_matrix = np.zeros((n_cells, n_cells), dtype=np.int8)
    for i, neighbors in enumerate(final_indices):
        if neighbors:
            adjacency_matrix[i, neighbors] = 1
            adjacency_matrix[neighbors, i] = 1
    return adjacency_matrix


def find_neighbors(adj_matrix: np.ndarray) -> list:
    """Extract neighbor list from adjacency matrix

    Args:
        adj_matrix: (N, N) Adjacency matrix

    Returns:
        Neighbor list
    """
    return [np.nonzero(row)[0].tolist() for row in adj_matrix]


def calculate_expression_rates(data_array: np.ndarray, neighbors_list: list) -> np.ndarray:
    """Calculate expression rates for each cell's neighbors

    Args:
        data_array: (N, G) Expression matrix
        neighbors_list: Neighbor list

    Returns:
        expression_rates: (N, G) Expression rate matrix
    """
    expression_rates = []
    for neighbors in neighbors_list:
        neighbor_data = data_array[neighbors] if neighbors else np.array([])
        if neighbor_data.size > 0:
            rate = np.mean(neighbor_data != 0, axis=0)
        else:
            rate = np.zeros(data_array.shape[1])
        expression_rates.append(rate)
    return np.vstack(expression_rates)


def keep_top_half_nonzero_per_row_vectorized(matrix: np.ndarray,
                                              threshold: float = 0.5) -> np.ndarray:
    """Keep top threshold proportion of non-zero elements with largest values per row

    Args:
        matrix: (N, G) Input matrix
        threshold: Retention proportion, default 0.5

    Returns:
        Processed matrix, non-retained elements set to zero
    """
    nonzero_counts = np.count_nonzero(matrix != 0, axis=1)
    num_to_keep = np.ceil(nonzero_counts * threshold).astype(int)
    mask = np.zeros_like(matrix, dtype=bool)
    for i in range(matrix.shape[0]):
        if nonzero_counts[i] > 0:
            nonzero_indices = np.nonzero(matrix[i])[0]
            sorted_nonzero_values = matrix[i][nonzero_indices]
            sorted_indices = np.argsort(-sorted_nonzero_values)[:num_to_keep[i]]
            mask[i, nonzero_indices[sorted_indices]] = True
    result = np.where(mask, matrix, 0)
    return result


def compute_imputation_mask(data: np.ndarray, pca_embeddings: np.ndarray,
                            n_pca_neighbors: int = 101, k_neighbors: int = 11,
                            mask_threshold: float = 0.5):
    """Compute imputation mask (full pipeline)

    Based on common neighbors of cosine and Euclidean distances, generate imputation mask
    through neighbor expression rates, marking zero-value positions that need imputation.

    Args:
        data: (N, G) Raw expression matrix (after normalization+log1p)
        pca_embeddings: (N, D) PCA embeddings
        n_pca_neighbors: Number of neighbors for PCA nearest neighbor search
        k_neighbors: Final number of neighbors to keep
        mask_threshold: Non-zero value retention proportion

    Returns:
        final_mask: (N, G) Imputation mask, 1 indicates needs imputation
        ori_mask: (N, G) Original zero-value mask, 1 indicates original data is 0
    """
    # Compute cosine and Euclidean nearest neighbors
    cosine_nn = NearestNeighbors(metric='cosine', algorithm='auto').fit(pca_embeddings)
    euclidean_nn = NearestNeighbors(metric='euclidean', algorithm='auto').fit(pca_embeddings)
    _, cosine_indices = cosine_nn.kneighbors(pca_embeddings, n_neighbors=n_pca_neighbors)
    _, euclidean_indices = euclidean_nn.kneighbors(pca_embeddings, n_neighbors=n_pca_neighbors)

    # Fuse neighbors
    final_indices, _ = get_final_neighbors(cosine_indices, euclidean_indices, k=k_neighbors)
    adj_matrix = build_adjacency_matrix_fast(final_indices)
    final_indices = find_neighbors(adj_matrix)

    # First round mask generation
    ori_mask = np.where(data == 0, 1, 0)
    exp_ratios = calculate_expression_rates(data, final_indices)
    exp_mask = keep_top_half_nonzero_per_row_vectorized(exp_ratios, mask_threshold)
    exp_mask = np.where(exp_mask > 0, 1, 0)
    final_mask = np.logical_and(exp_mask, ori_mask).astype(int)

    # Second round: Enhance using first round mask
    fake_data = data + final_mask
    exp_ratios = calculate_expression_rates(fake_data, final_indices)
    exp_mask = keep_top_half_nonzero_per_row_vectorized(exp_ratios, mask_threshold)
    exp_mask = np.where(exp_mask > 0, 1, 0)
    final_mask = np.logical_or(np.logical_and(exp_mask, ori_mask).astype(int), final_mask)

    return final_mask, ori_mask
