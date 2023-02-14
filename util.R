require(dplyr)
require(dirmult)
require(Matrix)

# ' SoftMax Operation
#' @param par vector
#' @return softmax(vector)
softmax <- function(par){
  n.par <- length(par)
  par1 <- sort(par, decreasing = TRUE)
  Lk <- par1[1]
  for (k in 1:(n.par-1)) {
    Lk <- max(par1[k+1], Lk) + log1p(exp(-abs(par1[k+1] - Lk))) 
  }
  val <- exp(par - Lk)
  return(val)
}

#' Perfect Binary Tree-based convolution matrix
#' @param D tree depth
#' @return S adjacency matrix (#leaves x #tree nodes)
pbt.adj <- function(D, .signed = FALSE) {
    if(D < 2) return(matrix(1))
    .N <- 2^D - 1
    .nodes <- 1:.N
    .leaves <- 1:(2^(D-1))
    .bot <- (2^(D-1)):.N
    .col <- .bot
    .row <- .leaves
    .elem <- rep(1, length(.leaves))
    for(d in seq(1,D-1)){
        .bot <- floor(.bot/2)
        if(.signed){
            .new.elem <- 2*ceiling(.leaves/2^(d-1)) %% 2 - 1
        } else {
            .new.elem <- rep(1, length(.leaves))
        }
        .elem <- c(.elem, .new.elem)
        .col <- c(.col, .bot)
        .row <- c(.row, .leaves)
    }
    sparseMatrix(i = .row,
                 j = .col,
                 x = .elem)
}


#' Simulate three based data
#' @param N #cells
#' @param G #genes
#' @param S #anchor genes per node
#' @param D_tree tree depth
#' @param gamma0
#' @param alpha0
#' @param data.file
#' @return list X (#cell x #genes), anchor_gene_idx, A (tree adjacency matrix) 

sim_data <- function(N, G, S, D_tree, data.file, gamma0 = 500, alpha0 = 5){
  
  if(file.exists(data.file)){
    return(readRDS(data.file))
  }
  
  # Construct adjacency matrix for tree with depth of Tr
  A <- pbt.adj(D_tree)
  dim(A)[1] -> T #leaves/topics 
  dim(A)[2] -> Tr #tree nodes

  # Sample gene selection probability vector \pi_{jg} from Dirichlet
  #gamma0 <- 500
  sample_name <- paste0("s", 1:N)
  gene_name <- paste0("g", 1:G)
  node_name <- paste0("tr", 1:Tr)
  topic_name <- paste0("t", 1:T)
  rownames(A) <- topic_name; colnames(A) <- node_name
  X <- matrix(0, nrow = N, ncol = G); rownames(X) <- sample_name; colnames(X) <- gene_name
  pi <- matrix(0,nrow = Tr, ncol = G); rownames(pi) <- node_name; colnames(pi) <- gene_name
  #node X gene
  anchor_gene_mat <- matrix(0, nrow = Tr, ncol = G) #node X gene, 0/1
  rownames(anchor_gene_mat) <- node_name; colnames(anchor_gene_mat) <- gene_name
  
  for(k in 1:Tr){
    pi_anchor <- rdirichlet(n=1, alpha=gamma0 * rep(1/S, S))
    anchor_gene_idx <- sample(G, S, replace=FALSE)
    anchor_gene_mat[k, anchor_gene_idx] <- 1
    pi[k, anchor_gene_idx] <- pi_anchor
  }

  # Sample node-specific effect size (take positive for simplicity)
  ## Normal(0, sd = layer * 0.5) + Normal(0, sd = 1), SNR increases with depth
  node_effect <- c()
  for(layer in 1:D_tree){
    node_effect[2^(layer-1):(2^layer-1)] <- rnorm(2^(layer-1), mean = 0, sd = layer * 0.5) + rnorm(2^(layer-1), mean = 0, sd = 1)
  }
  # Aggregate tree-node-specific topics to construct beta
  beta <- A %*% (node_effect * pi) #topic X gene
  # Sample topic proportions from Dirichlet
  theta <- rdirichlet(n = N, alpha = alpha0 * softmax(rnorm(T)))
  # Aggregate topic-specific gene activities
  rho_raw <- theta %*% beta
  rho <- apply(rho_raw, 1, softmax)
  rho <- t(rho)
  # Sample X from multinomial
  D <- exp(rnorm(N)) # sequencing depth from log-normal(0,1)
  for(i in 1:N){
    X[i,] <- rmultinom(1, round(D[i] * G), rho[i,])
  }
  param_list <- list(N = N, G = G, S = S,
                     D_tree = D_tree,
                     gamma0 = gamma0,
                     alpha0 = alpha0)

  res <- list(X = X, A = A, anchor_gene_mat = anchor_gene_mat,
              theta = theta,
              beta_node = pi, beta_topic = beta, 
              node_effect = node_effect,
              rho = rho, rho_raw = rho_raw,
              sequence_depth = D,
              param_list = param_list)
  saveRDS(res, data.file)
  return(res)
}
