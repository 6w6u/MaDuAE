from MaDuAEImpute import MaDuAEImputer

import pandas as pd
import scanpy as sc
import numpy as np

import os
import warnings
import scCAD

from collections import Counter
import matplotlib.pyplot as plt

# Set display for Chinese characters and minus signs
plt.rcParams['font.sans-serif'] = ['SimHei'] # Display Chinese characters
plt.rcParams['axes.unicode_minus'] = False # Display minus signs

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

my_palette = ["#000000", "#377EB8", "#4DAF4A", "#F781BF", "#FF7F00","#FFD92F",
                "#00FF00", "#E41A1C", "#984EA3", "#66C2A5", "#000080","#8DA0CB",
                "#FFFF33", "#A6D854"]

# ========== Define global configuration outside the loop ==========
RARE_CELL_TYPES = ["IMR90"] #Li:["IMR90"]
RARE_CELL_SET = set(RARE_CELL_TYPES)  # Accelerate lookup

def is_rare_label(label):
    """Check if it is a rare cell type"""
    return label in RARE_CELL_SET

def calculate_rare_cell_f1_simple(labels_true, clusters_result, cellNames, rare_labels):
    """
    Simplified version: All rare cells are merged for statistics, without distinguishing categories

    Parameters:
    -----------
    labels_true : list/array, True label for each cell
    clusters_result : list of list, Rare subpopulations returned by scCAD, each sublist is cell indices
    cellNames : list, Cell name/index list (strings)
    rare_labels : list, Rare cell type list (e.g., ["LEC 1", "HEC", ...])

    Returns:
    --------
    f1, precision, recall : float, Three metrics
    """
    import numpy as np
    from collections import Counter

    # 1. Mark all true rare cells (any rare type counts)
    rare_set = set(rare_labels)
    is_rare_true = np.array([lbl in rare_set for lbl in labels_true])
    true_rare_indices = set(np.where(is_rare_true)[0])
    n_true_rare = len(true_rare_indices)

    if n_true_rare == 0:
        return 0.0, 0.0, 0.0

    # 2. Collect all cell indices in rare subpopulations identified by scCAD (merge into one prediction set)
    predicted_rare_indices = set()
    for cluster_cells in clusters_result:
        # Convert cluster_cells to strings to match cellNames
        cluster_cells_str = [str(i) for i in cluster_cells]
        indices = np.where(np.isin(cellNames, cluster_cells_str))[0]
        predicted_rare_indices.update(indices)

    n_predicted = len(predicted_rare_indices)

    # 3. Calculate TP/FP/FN
    tp = len(true_rare_indices & predicted_rare_indices)  # Intersection: both rare and predicted
    fp = n_predicted - tp  # Predicted as rare but actually not
    fn = n_true_rare - tp  # Actually rare but not predicted

    # 4. Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall

# ============ Auxiliary function: Mark rare cells ============
def get_rare_cell_mask(labels_true, result, cellNames):
    """
    Generate rare cell boolean mask based on scCAD identification results

    Parameters:
    - result: List of rare subpopulations returned by scCAD, each element is a list of cell indices for that subpopulation
    """
    rare_mask = np.zeros(len(labels_true), dtype=bool)
    for cluster_cells in result:
        cluster_cells_str = [str(i) for i in cluster_cells]
        indices = np.where(np.isin(cellNames, cluster_cells_str))[0]
        rare_mask[indices] = True
    return rare_mask



#######################################################################################################################
filename = "Li"
savedir = filename+"_result/"
methodname = "MaDuAE"

data = pd.read_csv('Li_test.csv', index_col=0).T
df = pd.read_csv('Li_label.csv')
labels_true = df.iloc[:, -1].tolist()

data = data.astype(np.float32)
labels_true = np.array([str(i) for i in labels_true])

# 2. Label distribution statistics
print("=== Cell Label Distribution Statistics ===")
label_counts = Counter(labels_true)
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = count / len(labels_true) * 100
    print(f"{label}: {count} cells ({percentage:.2f}%)")
print(f"Total cells: {len(labels_true)}\n")

# 3. Run scCAD (without passing cellNames)
result, score, sub_clusters, degs_list = scCAD.scCAD(data=data,dataName=filename)

# Fix option 1: Convert cluster_cells to strings (recommended)
print("\n=== Rare Subpopulation Verification ===")
cellNames = [str(i) for i in range(data.shape[0])]

for cluster_idx, cluster_cells in enumerate(result):
    # Convert integer IDs to strings to match cellNames
    cluster_cells_str = [str(i) for i in cluster_cells]
    indices = np.where(np.isin(cellNames, cluster_cells_str))[0]

    if len(indices) > 0:
        print(f"\n=== Rare Subpopulation {cluster_idx + 1} ({len(indices)} cells) ===")
        print(f"True label distribution: {Counter(labels_true[indices])}")
    else:
        print(f"\n⚠️ Subpopulation {cluster_idx + 1}: No matching cells found")
# Calculate simplified F1
f1_pre, prec_pre, rec_pre = calculate_rare_cell_f1_simple(
    labels_true, result, cellNames, RARE_CELL_TYPES
)
print(f"\n🎯 Rare Cell Identification Evaluation (Merged Statistics {RARE_CELL_TYPES}):")
print(f"Before Imputation | Precision: {prec_pre:.4f} | Recall: {rec_pre:.4f} | F1: {f1_pre:.4f}")

# Rare cell mask before imputation (using result obtained from running scCAD on raw data)
rare_mask_before = get_rare_cell_mask(labels_true, result, cellNames)
n_rare_before = np.sum(rare_mask_before)
print(f"📊 Rare cells identified before imputation: {n_rare_before} / {len(labels_true)} "
      f"({n_rare_before/len(labels_true)*100:.2f}%)")

adata = sc.AnnData(data)
# Imputation
imputer = MaDuAEImputer()

adata_imputed = imputer.fit_transform(adata)

imp_result = adata_imputed.X
data = np.expm1(imp_result)# Use expm1 to reverse the log1p transformation.
# If gene and cell names are not provided, scCAD will generate them automatically.
result, score, sub_clusters, degs_list = scCAD.scCAD(data=data, dataName=filename)

# Fix option 1: Convert cluster_cells to strings (recommended)
print("\n=== Rare Subpopulation Verification ===")
cellNames = [str(i) for i in range(data.shape[0])]

for cluster_idx, cluster_cells in enumerate(result):
    # Convert integer IDs to strings to match cellNames
    cluster_cells_str = [str(i) for i in cluster_cells]
    indices = np.where(np.isin(cellNames, cluster_cells_str))[0]

    if len(indices) > 0:
        print(f"\n=== Rare Subpopulation {cluster_idx + 1} ({len(indices)} cells) ===")
        print(f"True label distribution: {Counter(labels_true[indices])}")
    else:
        print(f"\n⚠️ Subpopulation {cluster_idx + 1}: No matching cells found")
# Calculate simplified F1
f1_pre, prec_pre, rec_pre = calculate_rare_cell_f1_simple(
    labels_true, result, cellNames, RARE_CELL_TYPES
)
# Expect slight differences in results each time you run it.
print(f"\n🎯 Rare Cell Identification Evaluation (Merged Statistics {RARE_CELL_TYPES}):")
print(f"After Imputation | Precision: {prec_pre:.4f} | Recall: {rec_pre:.4f} | F1: {f1_pre:.4f}")
