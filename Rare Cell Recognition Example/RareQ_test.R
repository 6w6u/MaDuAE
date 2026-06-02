library(devtools)
install_github("fabotao/RareQ")
library(Seurat)
library(RareQ)
library(rhdf5)

################################################### 1%Jurkat ###################################################
hd5 <- h5read('1%Jurkat.h5','/') #
x1=data.frame(hd5$X)
rowname=as.vector(hd5$gn)
rownames(x1) <- rowname

t_labels=hd5$Y

sob.counts <- as.matrix(x1)

# Preprocess scRNA-seq data
sc_object <- CreateSeuratObject(counts = sob.counts, project = "sc_object", min.cells = 3)
sc_object <- AddMetaData(object = sc_object, metadata = t_labels, col.name = "celltype") 
lib_sizes <- colSums(GetAssayData(sc_object, slot = "counts"))
median_lib <- median(lib_sizes)
sc_object <- NormalizeData(sc_object, scale.factor = median_lib)

sc_object <- FindVariableFeatures(sc_object, nfeatures = 2000, selection.method = "disp")
sc_object <- ScaleData(sc_object)
sc_object <- RunPCA(sc_object, npcs = 50)
sc_object <- RunUMAP(sc_object, dims = 1:50)
sc_object <- FindNeighbors(
  object = sc_object,
  k.param = 20,
  compute.SNN = FALSE,
  prune.SNN = 0,
  reduction = "pca",
  dims = 1:50,
  force.recalc = FALSE,
  return.neighbor = TRUE
)

# Identify major and rare clusters
#cluster <- FindRare(imp_object)
cluster <- ConsensusRare(
  sc_object,
  assay = "RNA",
  reduction = "pca",
  dims = 1:50,
  k.param = 20,
  k = 6,
  Q_cut = 0.6,
  ratio = 0.2,
  reps = 30
)
table(cluster)

sc_object$cluster <- cluster
DimPlot(sc_object, group.by = "cluster")
DimPlot(sc_object, group.by = "celltype")

cm <- table(True = t_labels, Predicted = cluster)
print(cm)

cm_reordered <- reorder_for_diagonal(cm)
rownames(cm_reordered)[rownames(cm_reordered) == "jurkat"] <- "jurkat*"

# ===== 3. 计算百分比并绘图 =====
cm_percent <- prop.table(cm_reordered, margin = 1)
cm_long_pct <- as.data.frame(cm_percent)
cm_long_orig <- as.data.frame(cm_reordered)
cm_long <- cm_long_orig
cm_long$Percent <- cm_long_pct$Freq

g1 = ggplot(cm_long, aes(x = Predicted, y = True, fill = Percent)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq, color = Freq > 0), size = 10, fontface = "bold") +
  scale_fill_gradient(
    low = "#DAE3F5", high = "#4B0082", name = "Row %"
  ) +
  scale_color_manual(
    values = c("FALSE" = "black", "TRUE" = "white"),
    guide = "none"
  ) +
  scale_y_discrete(limits = rev) +
  labs(x = "RareQ Predicted", y = "True", 
       title = "Confusion Matrix (before imputation)") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(hjust = 0.5, size = 24, face = "bold"),
    axis.text.y = element_text(size = 24, face = "bold"),
    axis.title.x = element_text(size = 24, face = "bold"),
    axis.title.y = element_text(size = 24, face = "bold"),
    plot.title = element_text(size = 28, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 24, face = "bold"),
    legend.title = element_text(size = 24, face = "bold"),
    panel.grid = element_blank()
  )
g1

imp_data = read.csv("1%Jurkat_imputed.csv",row.names = 1,header = TRUE)
sob.counts <- expm1(as.matrix(imp_data))
sob.counts <- as(sob.counts, "sparseMatrix")

# Preprocess scRNA-seq data
imp_object <- CreateSeuratObject(counts = sob.counts, project = "sc_object", min.cells = 3)
imp_object <- AddMetaData(object = imp_object, metadata = t_labels, col.name = "celltype") 
lib_sizes <- colSums(GetAssayData(imp_object, slot = "counts"))
median_lib <- median(lib_sizes)
imp_object <- NormalizeData(imp_object, scale.factor = median_lib)

imp_object <- FindVariableFeatures(imp_object, nfeatures = 2000, selection.method = "disp")
imp_object <- ScaleData(imp_object)
imp_object <- RunPCA(imp_object, npcs = 50)
imp_object <- RunUMAP(imp_object, dims = 1:50)
imp_object <- FindNeighbors(
  object = imp_object,
  k.param = 20,
  compute.SNN = FALSE,
  prune.SNN = 0,
  reduction = "pca",
  dims = 1:50,
  force.recalc = FALSE,
  return.neighbor = TRUE
)

# Identify major and rare clusters
#cluster <- FindRare(imp_object)
cluster <- ConsensusRare(
  imp_object,
  assay = "RNA",
  reduction = "pca",
  dims = 1:50,
  k.param = 20,
  k = 6,
  Q_cut = 0.6,
  ratio = 0.2,
  reps = 30
)
table(cluster)

imp_object$cluster <- cluster
DimPlot(imp_object, group.by = "cluster")
DimPlot(imp_object, group.by = "celltype")

cm <- table(True = t_labels, Predicted = cluster)
print(cm)

cm_reordered <- reorder_for_diagonal(cm)
rownames(cm_reordered)[rownames(cm_reordered) == "jurkat"] <- "jurkat*"

cm_percent <- prop.table(cm_reordered, margin = 1)
cm_long_pct <- as.data.frame(cm_percent)
cm_long_orig <- as.data.frame(cm_reordered)
cm_long <- cm_long_orig
cm_long$Percent <- cm_long_pct$Freq

g2 = ggplot(cm_long, aes(x = Predicted, y = True, fill = Percent)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq, color = Freq > 0), size = 10, fontface = "bold") +
  scale_fill_gradient(
    low = "#DAE3F5", high = "#4B0082", name = "Row %"
  ) +
  scale_color_manual(
    values = c("FALSE" = "black", "TRUE" = "white"),
    guide = "none"
  ) +
  scale_y_discrete(limits = rev) +
  labs(x = "RareQ Predicted", y = "True", 
       title = "Confusion Matrix (after imputation)") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(hjust = 0.5, size = 24, face = "bold"),
    axis.text.y = element_text(size = 24, face = "bold"),
    axis.title.x = element_text(size = 24, face = "bold"),
    axis.title.y = element_text(size = 24, face = "bold"),
    plot.title = element_text(size = 28, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 24, face = "bold"),
    legend.title = element_text(size = 24, face = "bold"),
    panel.grid = element_blank()
  )
g2
################################################### Li ###################################################

sc_data = read.csv("Li_test.csv",row.names = 1,header = TRUE)
sob.counts <- as.matrix(sc_data)
sob.counts <- as(sob.counts, "sparseMatrix")

t_labels = read.csv("Li_label.csv",row.names = 1,header = TRUE)
t_labels = unlist(t_labels$hd5.Y)

# Preprocess scRNA-seq data
sc_object <- CreateSeuratObject(counts = sob.counts, project = "sc_object", min.cells = 3)
sc_object <- AddMetaData(object = sc_object, metadata = t_labels, col.name = "celltype") 
lib_sizes <- colSums(GetAssayData(sc_object, slot = "counts"))
median_lib <- median(lib_sizes)
sc_object <- NormalizeData(sc_object, scale.factor = median_lib)

sc_object <- FindVariableFeatures(sc_object, nfeatures = 2000, selection.method = "disp")
sc_object <- ScaleData(sc_object)
sc_object <- RunPCA(sc_object, npcs = 50)
sc_object <- RunUMAP(sc_object, dims = 1:50)
sc_object <- FindNeighbors(
  object = sc_object,
  k.param = 20,
  compute.SNN = FALSE,
  prune.SNN = 0,
  reduction = "pca",
  dims = 1:50,
  force.recalc = FALSE,
  return.neighbor = TRUE
)

# Identify major and rare clusters
#cluster <- FindRare(imp_object)
cluster <- ConsensusRare(
  sc_object,
  assay = "RNA",
  reduction = "pca",
  dims = 1:50,
  k.param = 20,
  k = 6,
  Q_cut = 0.6,
  ratio = 0.2,
  reps = 30
)
table(cluster)

sc_object$cluster <- cluster
DimPlot(sc_object, group.by = "cluster")
DimPlot(sc_object, group.by = "celltype")

cm <- table(True = t_labels, Predicted = cluster)
print(cm)

cm_reordered <- reorder_for_diagonal(cm)
rownames(cm_reordered)[rownames(cm_reordered) == "IMR90"] <- "IMR90*"

cm_percent <- prop.table(cm_reordered, margin = 1)
cm_long_pct <- as.data.frame(cm_percent)
cm_long_orig <- as.data.frame(cm_reordered)
cm_long <- cm_long_orig
cm_long$Percent <- cm_long_pct$Freq

g3 = ggplot(cm_long, aes(x = Predicted, y = True, fill = Percent)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq, color = Freq > 0), size = 10, fontface = "bold") +
  scale_fill_gradient(
    low = "#DAE3F5", high = "#4B0082", name = "Row %"
  ) +
  scale_color_manual(
    values = c("FALSE" = "black", "TRUE" = "white"),
    guide = "none"
  ) +
  scale_y_discrete(limits = rev) +
  labs(x = "RareQ Predicted", y = "True", 
       title = "Confusion Matrix (before imputation)") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(hjust = 0.5, size = 24, face = "bold"),
    axis.text.y = element_text(size = 24, face = "bold"),
    axis.title.x = element_text(size = 24, face = "bold"),
    axis.title.y = element_text(size = 24, face = "bold"),
    plot.title = element_text(size = 28, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 24, face = "bold"),
    legend.title = element_text(size = 24, face = "bold"),
    panel.grid = element_blank()
  )
g3

imp_data = read.csv("Li_impute_final.csv",row.names = 1,header = TRUE)
sob.counts <- expm1(as.matrix(imp_data))
sob.counts <- as(sob.counts, "sparseMatrix")

# Preprocess scRNA-seq data
imp_object <- CreateSeuratObject(counts = sob.counts, project = "sc_object", min.cells = 3)
imp_object <- AddMetaData(object = imp_object, metadata = t_labels, col.name = "celltype")

lib_sizes <- colSums(GetAssayData(imp_object, slot = "counts"))
median_lib <- median(lib_sizes)
imp_object <- NormalizeData(imp_object, scale.factor = median_lib)

imp_object <- FindVariableFeatures(imp_object, nfeatures = 2000, selection.method = "disp")
imp_object <- ScaleData(imp_object)
imp_object <- RunPCA(imp_object, npcs = 50)
imp_object <- RunUMAP(imp_object, dims = 1:50)
imp_object <- FindNeighbors(
  object = imp_object,
  k.param = 20,
  compute.SNN = FALSE,
  prune.SNN = 0,
  reduction = "pca",
  dims = 1:50,
  force.recalc = FALSE,
  return.neighbor = TRUE
)

# Identify major and rare clusters
#cluster <- FindRare(imp_object)
cluster <- ConsensusRare(
  imp_object,
  assay = "RNA",
  reduction = "pca",
  dims = 1:50,
  k.param = 20,
  k = 6,
  Q_cut = 0.6,
  ratio = 0.2,
  reps = 30
)
table(cluster)

imp_object$cluster <- cluster
DimPlot(imp_object, group.by = "cluster")
DimPlot(imp_object, group.by = "celltype")

cm <- table(True = t_labels, Predicted = cluster)
print(cm)

cm_reordered <- reorder_for_diagonal(cm)
rownames(cm_reordered)[rownames(cm_reordered) == "IMR90"] <- "IMR90*"

cm_percent <- prop.table(cm_reordered, margin = 1)
cm_long_pct <- as.data.frame(cm_percent)
cm_long_orig <- as.data.frame(cm_reordered)
cm_long <- cm_long_orig
cm_long$Percent <- cm_long_pct$Freq

g4 = ggplot(cm_long, aes(x = Predicted, y = True, fill = Percent)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq, color = Freq > 0), size = 10, fontface = "bold") +
  scale_fill_gradient(
    low = "#DAE3F5", high = "#4B0082", name = "Row %"
  ) +
  scale_color_manual(
    values = c("FALSE" = "black", "TRUE" = "white"),
    guide = "none"
  ) +
  scale_y_discrete(limits = rev) +
  labs(x = "RareQ Predicted", y = "True", 
       title = "Confusion Matrix (after imputation)") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(hjust = 0.5, size = 24, face = "bold"),
    axis.text.y = element_text(size = 24, face = "bold"),
    axis.title.x = element_text(size = 24, face = "bold"),
    axis.title.y = element_text(size = 24, face = "bold"),
    plot.title = element_text(size = 28, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 24, face = "bold"),
    legend.title = element_text(size = 24, face = "bold"),
    panel.grid = element_blank()
  )
g4


