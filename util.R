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
#' @param g #anchor genes
#' @param D_tree tree depth
#' @param gamma0 
#' @param alpha0
#' @param data.file
#' @return list X (#cell x #genes), anchor_gene_idx, A (tree adjacency matrix) 

sim_data <- function(N, G, g, D_tree, data.file, gamma0 = 50, alpha0 = 5){
  
  if(file.exists(data.file)){
    return(readRDS(data.file))
  }
  # Construct adjacency matrix for tree with depth of Tr
  A <- pbt.adj(D_tree)
  dim(A)[1] -> T #leaves/topics 
  dim(A)[2] -> Tr #tree nodes
  # Sample gene selection probability vector \pi_{jg} from Dirichlet
  #gamma0 <- 50
  pi <- rdirichlet(n=Tr, alpha=gamma0 * rep(1/g, g))
  # Sample node-specific effect size (positive)
  beta_node <- rgamma(n=Tr, shape=1)
  # Aggregate tree-node-specific topics to construct beta
  beta <- A %*% (beta_node * pi)
  # Sample topic proportions from Dirichlet
  #alpha0 <- 5
  theta <- rdirichlet(n = N, alpha = alpha0 * rep(1/T, T))
  # Aggregate topic-specific gene activities
  rho <- theta %*% beta
  # Sample X from multinomial
  D <- rbeta(N, 1,1) # sequencing depth
  X_anchor_gene <- matrix(0, nrow=N, ncol=g)
  for(i in 1:N){   
    X_anchor_gene[i,] <- rmultinom(1, g, D[i] *  softmax(rho[i,]))
  }
  # Assign the anchor genes to X
  X <- matrix(0, nrow=N, ncol=G)
  anchor_gene_idx <- sample(G, g, replace=FALSE)
  for(j in g){
    X[,anchor_gene_idx[j]] <- X_anchor_gene[,j] 
  }
  
  res <- list(X = X, anchor_gene_idx = anchor_gene_idx, A = A)
  saveRDS(res, data.file)
  return(res)
}