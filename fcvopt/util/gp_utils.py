import numpy as np
from scipy.linalg import cholesky,solve_triangular

def kernel_inv(self,kernel,X,det=True):
    K = kernel(X)
    K[np.diag_indices_from(K)] += self.eps
    L = cholesky(K, lower=True)  # Line 2
    
    L_inv = solve_triangular(L,np.eye(L.shape[0]),lower=True)
    K_inv = L_inv.T.dot(L_inv)
    
    if det:
        ldet_K = 2*np.sum(np.log(np.diag(L)))
        return K_inv,ldet_K
    else:
        return K_inv