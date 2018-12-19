# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize,Bounds

from samplers import uniform_sampler
#from fcvopt.util.samplers import uniform_sampler

def scipy_minimize(fun,lower,upper,n_restarts=10,maximize=False,rng=None):
    mstarts = uniform_sampler(n_restarts,lower,upper,rng)
    x_best = np.copy(lower)
    f_best = np.Inf
    
    for i in range(n_restarts):
        res = minimize(fun,mstarts[i,:],method='L-BFGS-B',
                       bounds=Bounds(lower,upper))
        if res.fun < f_best:
            x_best = res.x
            f_best = res.fun
            
    return x_best,f_best
        
              
    
        
