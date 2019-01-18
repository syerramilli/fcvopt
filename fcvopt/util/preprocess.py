# -*- coding: utf-8 -*-

import numpy as np

def zero_one_scale(X,lower=None,upper=None):
    if lower is None:
        lower = np.min(X,axis=0)
        
    if upper is None:
        upper = np.max(X,axis=0)
    
    X_scaled = (X-lower)/(upper-lower)
    return X_scaled,lower,upper
    
def zero_one_rescale(X_scaled,lower,upper):
    return lower + (upper-lower)*X_scaled.copy()

def standardize_vec(v):
    loc = np.mean(v)
    scale = np.std(v)
    return (v-np.mean(v))/np.std(v),loc,scale

def unstandardize_vec(v,loc,scale):
    return v*scale + loc