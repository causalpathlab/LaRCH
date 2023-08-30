library(umap)
library(ggplot2)
library(patchwork)
library(rsvd)
library(ROCR)
library(clue)
library(data.table)
library(anndata)
library(ComplexHeatmap)
library(data.table)
source("util.R")

# load simulated data
sim_seed <- 123
sim_data_file <- paste0("data/sim_layer4_seed", sim_seed, ".h5ad")

if (file.exists(sim_data_file)) {
  simulated_data <- list()
  ad <- read_h5ad(sim_data_file)
  expression_matrix <- as.matrix(data.frame(ad))
  simulated_data$expression_matrix <- expression_matrix
  simulated_data$gene_group <- ad$var$gene_group
  rm(ad)
}else {
  # simulate single cell data by layer
  set.seed(123)
  simulated_data_layer1 <- simulate_single_cell_data(n_cells = 2000, n_genes = 500, cluster_ratios = c(1))
  simulated_data_layer2 <- simulate_single_cell_data(n_cells = 2000, n_genes = 500, cluster_ratios = runif(2))
  simulated_data_layer3 <- simulate_single_cell_data(n_cells = 2000, n_genes = 500, cluster_ratios = runif(4))
  simulated_data_layer4 <- simulate_single_cell_data(n_cells = 2000, n_genes = 500, cluster_ratios = runif(8))

  simulated_data <- list()
  simulated_data$expression_matrix <- rbind(simulated_data_layer1$expression_matrix, simulated_data_layer2$expression_matrix,simulated_data_layer3$expression_matrix, simulated_data_layer4$expression_matrix)

  simulated_data$gene_group <- factor(c(paste0("a", simulated_data_layer1$group_ids), paste0("b", simulated_data_layer2$group_ids), paste0("c", simulated_data_layer3$group_ids), paste0("d", simulated_data_layer4$group_ids)))

  # Extract expression matrix from simulated data
  expression_matrix <- simulated_data$expression_matrix # nolint

  ad <- AnnData(X = t(expression_matrix), 
                obs = data.frame(row.names = paste0("cell", 1:ncol(expression_matrix))),
                var = data.frame(gene_group = simulated_data$gene_group, 
                                row.names = paste0("gene", 1:nrow(expression_matrix)))
                )

  write_h5ad(ad, sim_data_file)
}


# Perform UMAP on the gene
set.seed(123)
umap_result_cell <- umap(t(expression_matrix))
umap_result_gene <- umap(expression_matrix)

df_umap_gene <- data.frame(
  UMAP1 = umap_result_gene$layout[, 1],
  UMAP2 = umap_result_gene$layout[, 2],
  gene_group = simulated_data$gene_group
)

df_umap_cell <- data.frame(
  UMAP1 = umap_result_cell$layout[, 1],
  UMAP2 = umap_result_cell$layout[, 2]
)

# UMAP plots
umap_plot_gene <- ggplot(df_umap_gene, aes(x = UMAP1, y = UMAP2, color = gene_group)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "UMAP projection of Genes") +
  scale_color_discrete(name = "Gene Group")


umap_plot_cell <- ggplot(df_umap_cell, aes(x = UMAP1, y = UMAP2)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "UMAP projection of Cells")

p_umap <- umap_plot_gene + umap_plot_cell
ggsave("plots/umap.pdf", p_umap, width = 10, height = 5)

# Randomized SVD

K <- 15 # 1 + 2 + 4 + 8

# Compute AUC for each PC-gene group pair
auc_matrix <- matrix(0, nrow = K, ncol = length(unique(simulated_data$gene_group)))

for (i in 1:K) {
  for (j in seq_along(unique(simulated_data$gene_group))) {
    gene_group_id <- unique(simulated_data$gene_group)[j]
    actual_genes <- which(simulated_data$gene_group == gene_group_id)
    # Create a binary vector: 1 if gene is in the actual group, 0 otherwise
    truth <- ifelse(1:nrow(expression_matrix) %in% actual_genes, 1, 0)
    
    ## Predicted scores from the PC
    # set.seed(123)
    # rsvd_result <- rsvd(expression_matrix, k = K)
    # scores <- rsvd_result$u[, i]
    ## Predicted score from the tree model
    tree_weight <- weight.mat
    scores <- tree_weight[i, ]
    pred <- prediction(scores, truth)
    perf <- performance(pred, measure = "aucpr")
    auc_matrix[i, j] <- as.numeric(perf@y.values)
  }
}

# Match PCs with gene groups using Hungarian algorithm
cost_matrix <- 1 - auc_matrix  # Convert AUC to a cost (1 - AUC)
hungarian_result <- solve_LSAP(cost_matrix)

matched_pairs <- data.table(
  GeneGroup = unique(simulated_data$gene_group)[hungarian_result],
  PC = 1:K,
  Matched_AUC = auc_matrix[cbind(1:K, hungarian_result)]
)
setorder(matched_pairs, GeneGroup)
matched_pairs
Heatmap(auc_matrix)