simulate_single_cell_data <- function(n_cells, n_genes, cluster_ratios = NULL) {
  # Ensure the necessary packages are loaded
  if (!require(Matrix)) {
    install.packages("Matrix")
    library(Matrix)
  }
  
  # Compute genes per group
  cluster_ratios = cluster_ratios / sum(cluster_ratios)
  genes_per_group <- round(n_genes * cluster_ratios)
  
  # Distribute any leftover genes due to rounding among the groups
  leftover_genes <- n_genes - sum(genes_per_group)
  for (i in 1:leftover_genes) {
    genes_per_group[i] <- genes_per_group[i] + 1
  }
  
  # Placeholder for expression matrix
  expression_matrix <- Matrix(0, nrow=n_genes, ncol=n_cells, sparse=TRUE)
  
  gene_ids <- vector("character", n_genes)
  group_ids <- integer(n_genes)
  current_gene_idx <- 1

  # Generate data per gene group
  for (group in seq_along(cluster_ratios)) {
    # Set the range for genes in this group
    gene_range <- current_gene_idx:(current_gene_idx + genes_per_group[group] - 1)
    
    # Avoid index overflow
    if (max(gene_range) > n_genes) {
      gene_range <- current_gene_idx:n_genes
    }
    
    # Generate group-specific average expression and dispersions
    avg_expression <- rgamma(n_cells, shape=1, scale=5) + rnorm(n_cells, mean=0, sd=2*group)
    dispersions <- rgamma(n_cells, shape=0.5, scale=2)
    
    # Ensure no dispersion value is too close to zero
    dispersions[dispersions < 1e-5] <- 1e-5
    
    # Ensure average expression is non-negative
    avg_expression[avg_expression < 0] <- 0
    
    for (cell in 1:n_cells) {
      mu <- avg_expression[cell]
      theta <- 1/dispersions[cell]
      
      expression_matrix[gene_range, cell] <- rnbinom(length(gene_range), mu=mu, size=theta)
    }
    
    gene_ids[gene_range] <- paste0("Gene_", gene_range)
    group_ids[gene_range] <- group
    
    current_gene_idx <- max(gene_range) + 1
  }
  
  cell_ids <- paste0("Cell_", 1:n_cells)
  
  return(list(
    expression_matrix = expression_matrix,
    gene_ids = gene_ids,
    cell_ids = cell_ids,
    group_ids = group_ids
  ))
}