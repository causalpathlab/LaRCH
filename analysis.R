library(aricode)
library(data.table)
library(dplyr)
source("util.R")

#model_path <- "models/tree_spike_slab_ep2000_treeD5_bs128_lr0.01_train_size1_pip0.1_klbeta1_seed66"
#model_path <- "models/tree_spike_slab_ep1000_treeD5_bs128_lr0.01_train_size1_pip0.1_kl10.0_klbeta1_seed66"
#model_path <- "models/tree_spike_slab_ep500_treeD5_bs128_lr0.01_train_size1_pip0.1_kl1_klbeta1_seed66"

#model_path <- "models/tree_spike_slab_ep200_treeD5_bs128_lr0.01_train_size1_pip0.1_kl1_klbeta1_seed66"
#model_path <- "models/tree_spike_slab_ep100_treeD5_bs128_lr0.01_train_size1_pip0.1_kl10.0_klbeta1_seed66"
#model_path <- "models/tree_spike_slab_ep500_treeD5_bs128_lr0.01_train_size1_pip1.0_kl10.0_klbeta1_seed66"
#model_path <- "models/tree_spike_slab_ep200_treeD10_bs128_lr0.01_train_size1_pip1.0_kl10.0_klbeta1_seed66"
#model_path <- "models/tree_spike_slab_ep200_treeD3_bs128_lr0.01_train_size1_pip1.0_kl10.0_klbeta1_seed66"

#model_path <- "models/tree_spike_slab_ep200_treeD5_bs128_lr0.01_train_size1_pip1.0_kl0.1_klbeta1_seed66"
model_path <-"models/tree_spike_slab_ep200_treeD5_bs128_lr0.01_train_size1_pip0.1_kl0.1_klbeta1_seed66"
topics_est <- as.matrix(fread(paste0(model_path, "/topics.csv")), rownames = 1)

apply(topics_est, 1, which.max) %>% table()
heatmap(topics_est)
dat <- readRDS("data/sim_tree.rda")
apply(dat$theta, 1, softmax) -> theta

apply(theta, 2, which.max) %>% table()
t(theta) %>% heatmap()


### Normalized Mutual Information
NMI(paste0("topic", c(apply(topics_est, 1, which.max))), c(t(apply(theta, 2, which.max))))

data(iris)
cl <- cutree(hclust(dist(iris[,-5])), 4)
NID(cl, iris$Species)
####
logs <- fread("logs/tree_spike_slab_ep2000_treeD5_bs128_lr0.01_train_size1_pip0.1_klbeta1_seed66/20230110/metrics.csv")
logs %>% select("elbo_train", "train_loss_epoch", "epoch", "kl_beta_train", "kl_local_train", "reconstruction_loss_train") %>% na.omit() -> logs

plot(logs$elbo_train ~ logs$epoch)
plot(logs$train_loss_epoch ~ logs$epoch)
plot(logs$kl_beta_train ~ logs$epoch)
plot(logs$kl_local_train ~ logs$epoch)
plot(logs$reconstruction_loss_train ~ logs$epoch)