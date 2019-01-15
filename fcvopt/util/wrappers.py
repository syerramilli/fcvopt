# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize,Bounds

from fcvopt.util.samplers import lh_sampler

def scipy_minimize(fun,lower,upper,n_restarts=5,maximize=False,rng=None):
    mstarts = lh_sampler(n_restarts+1,lower,upper,rng)
    x_best = np.copy(lower)
    f_best = np.Inf
    
    for i in range(n_restarts+1):
        res = minimize(fun,mstarts[i,:],method='L-BFGS-B',
                       bounds=Bounds(lower,upper))
        if res.fun < f_best:
            x_best = res.x
            f_best = res.fun
            
    return x_best,f_best