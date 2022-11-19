import numpy as np
import scipy


np.random.seed(123)

tau = 1
k = 16
genes = 1000
cells = 100
reads = 1500

def generate_dense(tau, k, genes, cells, reads):

    beta = np.random.normal(size = (k, genes))

    delta = np.random.multivariate_normal(np.zeros(k), np.identity(k), size = cells)
    theta = scipy.special.softmax(delta, axis = 1)

    print(beta.shape)
    print(delta.shape)
    print(theta.shape)

    print(theta[1:10,:])
    print(np.sum(theta[1:10,:], axis = 1))

    rho_tilde = np.exp(np.matmul(theta, beta))

    print(rho_tilde.shape)
    print(rho_tilde[1:10, 1:10])

    rho = np.apply_along_axis(np.random.dirichlet, 1, rho_tilde)

    print(rho.shape)
    print(rho[1:10, 1:10])
    print(np.sum(rho[1:10,:], axis = 1))

    X = np.apply_along_axis(lambda a: np.random.multinomial(reads, a), 1, rho)

    print(X.shape)
    print(X[1:10, 1:10])
    print(np.sum(X[1:10,:], axis = 1))

    return X
