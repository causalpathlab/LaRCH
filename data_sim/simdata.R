source("util.R")
library(anndata)

seed <- 123
set.seed(seed)

N <- 5000
G <- 10000
S <- 50
D_tree <- 4 # 5
dat <- sim_data(N, G, S, D_tree, "data/sim_tree.rda")

## output a h5ad
ad <- AnnData(X = dat$X,
              obs = data.frame(sample_id = rownames(dat$X)),
              var = data.frame(gene = colnames(dat$X)))

ad$write_h5ad(filename = "data/sim_tree.h5ad")
