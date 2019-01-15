# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize,Bounds

from fcvopt.util.samplers import uniform_sampler

def scipy_minimize(fun,x0,lower,upper,n_restarts=1,maximize=False,rng=None):
    mstarts = np.concatenate((x0.reshape((1,-1)),
                              uniform_sampler(n_restarts,lower,upper,rng)),
                                axis=0)
    x_best = np.copy(lower)
    f_best = np.Inf
    
    for i in range(n_restarts+1):
        res = minimize(fun,mstarts[i,:],method='L-BFGS-B',
                       bounds=Bounds(lower,upper))
        if res.fun < f_best:
            x_best = res.x
            f_best = res.fun
            
    return x_best,f_best