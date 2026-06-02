import pandas as pd
import numpy as np
import scanpy as sc
from MaDuAEImpute import MaDuAEImputer

# Load data
data = pd.read_csv("iLNs_test.csv", encoding="utf-8", index_col=0).T
adata = sc.AnnData(data)

# Load true labels
df_labels = pd.read_csv('iLNs_label.csv')
labels_true = df_labels.iloc[:, -1].tolist()
label_to_int = {label: i for i, label in enumerate(set(labels_true))}
true_labels = np.array([label_to_int[label] for label in labels_true])

# Imputation
imputer = MaDuAEImputer(
    hidden_dim=32,
    num_epochs=300,
    learning_rate=1e-3,
    weight_decay=3e-4,
    couple_weight=1.0,
    consist_weight=0.1,
    inter_tower_sim_weight=0.1,
    patience=15,
    batch_size=1024,
)

adata_imputed = imputer.fit_transform(
    adata,
    true_labels=true_labels
    #save_best_model="iLNs_MaDuAE.pth"
)
'''
# Save imputation results
pd.DataFrame(
    adata_imputed.X.T,
    index=adata_imputed.var.index,
    columns=adata_imputed.obs.index
).to_csv("iLNs_MaDuAE_impute.csv")
'''
results = []
# Record results
if 'clustering_metrics' in adata_imputed.uns:
    metrics = adata_imputed.uns['clustering_metrics']
    results.append({
        'dataset': 'iLNs',
        'ari_before': metrics['ari_before'],
        'ari_after': metrics['ari_after'],
        'ari_improvement': metrics['ari_improvement'],
        'nmi_before': metrics['nmi_before'],
        'nmi_after': metrics['nmi_after'],
        'nmi_improvement': metrics['nmi_improvement'],
    })

# Print summary results
print("\n===== Summary Results =====")
for r in results:
    print(f"{r['dataset']}: ARI {r['ari_before']:.4f} -> {r['ari_after']:.4f} "
          f"(+{r['ari_improvement']:.4f})")