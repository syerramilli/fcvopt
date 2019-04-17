import numpy as np
from scipy.linalg import cholesky,cho_solve

def kernel_inv(kernel,X,eps,det=True):
    K = kernel(X)
    if X.shape[0] == 1:
        "If only 1 observation, there is no need to compute cholesky"
        K_inv = 1/K
        
        if det:
            ldet_K = np.log(K[0,0])
            return K_inv,ldet_K
        else:
            return K_inv
        
    K[np.diag_indices_from(K)] += eps*K[0,0]
    U = cholesky(K,check_finite=False)
    K_inv = cho_solve((U,False),np.eye(U.shape[0]),check_finite=False)
    
    if det:
        ldet_K = 2*np.sum(np.log(np.diag(U)))
        return K_inv,ldet_K
    else:
        return K_inv