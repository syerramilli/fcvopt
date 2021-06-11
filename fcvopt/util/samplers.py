# -*- coding: utf-8 -*-

import numpy as np

def uniform_sampler(n_samples,lower,upper,rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(20000))
        
    n_dims = lower.shape[0]
    
    # generate
    samples = np.stack([rng.uniform(lower[i],upper[i],n_samples) \
                     for i in np.arange(n_dims)],axis=1)
    return samples
    

def lh_sampler(n_samples,lower,upper,rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(20000))
        
    n_dim = lower.shape[0]
    
    # generate row and column grids
    grid_bounds = np.stack([np.linspace(lower[i], upper[i], n_samples+1) \
                            for i in np.arange(n_dim)],axis=1)
    grid_lower = grid_bounds[:-1,:]
    grid_upper = grid_bounds[1:,:]
    
    # generate 
    grid = grid_lower + (grid_upper-grid_lower)*rng.rand(n_samples,n_dim)
    
    # shuffle and return
    for i in range(n_dim):
        rng.shuffle(grid[:,i])
    
    return grid

def stratified_sample(num_choices:int,size:int) -> np.ndarray:
    num_mult = size//num_choices
    idx = np.repeat(np.arange(num_choices),num_mult)
    num_balance = size - num_mult*num_choices
    if num_balance > 0:
        idx = np.concatenate([idx,np.random.choice(num_choices,size=num_balance,replace=False)])

    np.random.shuffle(idx)
    return idx