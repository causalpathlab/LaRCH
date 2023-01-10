source("util.R")
library(anndata)
N <- 5000
G <- g <- 500
D_tree <- 5
dat <- sim_data(N, G, g, D_tree, "data/out.rda")

#######
ad <- AnnData(X = dat$X,
              obs = data.frame(sample_id = paste0("cell", 1:nrow(dat$X))),
              var = data.frame(gene = paste0("gene", 1:ncol(dat$X))))

ad$write_h5ad(filename = "data/sim_tree.h5ad")
