source("util.R")
library(anndata)

seed <- 123
set.seed(seed)

N <- 5000
G <- g <- 1000
D_tree <- 5
dat <- sim_data(N, G, g, D_tree, "data/sim_tree.rda")

## output a h5ad
##TODO change dataloader for online trainning
ad <- AnnData(X = dat$X_anchor_gene,
              obs = data.frame(sample_id = paste0("cell", 1:nrow(dat$X))),
              var = data.frame(gene = paste0("gene", 1:ncol(dat$X))))

ad$write_h5ad(filename = "data/sim_tree.h5ad")
