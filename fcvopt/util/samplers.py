# -*- coding: utf-8 -*-

import numpy as np


def uniform_sampler(n_samples,lower,upper,rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(20000))
        
    n_dims = lower.shape[0]
    
    # generate
    samples = np.stack((rng.uniform(lower[i],upper[i],n_samples) \
                     for i in range(n_dims)),axis=1)
    return samples
    

def lh_sampler(n_samples,lower,upper,rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(20000))
        
    n_dim = lower.shape[0]
    
    # generate row and column grids
    grid_bounds = np.stack((np.linspace(lower[i], upper[i], n_samples+1) \
                            for i in range(2)),axis=1)
    grid_lower = grid_bounds[:-1,:]
    grid_upper = grid_bounds[1:,:]
    
    # generate 
    grid = grid_lower + (grid_upper-grid_lower)*rng.rand(n_samples,n_dim)
    
    # shuffle and return
    for i in range(n_dim):
        rng.shuffle(grid[:,i])
    
    return grid
    