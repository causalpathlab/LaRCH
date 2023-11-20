suppressMessages(library(data.table))
suppressMessages(library(zellkonverter))
suppressMessages(library(SingleCellExperiment))
suppressMessages(library(scater))
suppressMessages(library(scran))
suppressMessages(library(bluster))
suppressMessages(library(aricode))
suppressMessages(library(tidyverse))

args = commandArgs(trailingOnly = TRUE)
print(args)
sc_data_file <- args[1]
topics_file <- args[2]
outfile <- args[3]

seed <- strsplit(sc_data_file, split = "seed")[[1]][2] 
seed <- strsplit(seed, split = "[.]")[[1]][1]
noise <- strsplit(sc_data_file, split = "rho")[[1]][3]
noise <- strsplit(noise, split = "_")[[1]][1]

sc_data <- readH5AD(sc_data_file)

sc_data <- logNormCounts(sc_data, assay.type = "X")
sc_data <- runPCA(sc_data)
sc_data <- runUMAP(sc_data)

topics <- fread(topics_file) %>%
  remove_rownames() %>% column_to_rownames("V1")

n_topics <- ncol(topics)
topic_factors <- factor(
  paste0('topic_', 0:(n_topics - 1)), 
  levels =  paste0('topic_', 0:(n_topics - 1)))

sc_data$max_topic <- topic_factors[apply(topics, 1, which.max)]

print("Clustering...")

reducedDim(sc_data, "topic_space") <- topics %>% select_if(is.numeric)

PCA_clusters <- clusterCells(sc_data, use.dimred = "PCA", BLUSPARAM=NNGraphParam(cluster.fun="louvain"))
sc_data$PCA_clusters <- PCA_clusters

UMAP_clusters <- clusterCells(sc_data, use.dimred = "UMAP", BLUSPARAM=NNGraphParam(cluster.fun="louvain"))
sc_data$UMAP_clusters <- UMAP_clusters

z_clusters <- clusterCells(sc_data, use.dimred = "topic_space", BLUSPARAM=NNGraphParam(cluster.fun="louvain"))
sc_data$z_clusters <- z_clusters

NMI_df <- data.frame(
  labels = c("max_topic", "LaRCH Cluster", "PCA Cluster", "UMAP Cluster"), 
  NMI = c(NMI(sc_data$cell_type, sc_data$max_topic),
          NMI(sc_data$cell_type, sc_data$z_clusters), 
          NMI(sc_data$cell_type, sc_data$PCA_clusters),
          NMI(sc_data$cell_type, sc_data$UMAP_clusters)
          )
)

NMI_df$noise <- noise
NMI_df$seed <- seed

print("writing NMI values to outfile")
write.table(NMI_df, 
            file = outfile,
            sep = ",", 
            row.names = FALSE,
            col.names = FALSE
            )


