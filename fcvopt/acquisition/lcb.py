# -*- coding: utf-8 -*-

import numpy as np

class LCB:
    def __init__(self,model,kappa=2):
        self.model = model
        self.kappa = kappa
        
    def update(self,model):
        self.model = model
        
    def __call__(self,x):
        x_copy = np.copy(x)
        if x.ndim == 1:
            x_copy = x_copy.reshape((1,-1))
        y_mean,y_std = self.model.predict(x_copy,scaled=True,return_std=True)
        val = y_mean - self.kappa*y_std
        return val
    

class LCBMCMC:
    def __init__(self,model_mcmc,kappa=2):
        self.acq = [LCB(model,kappa) for model in model_mcmc.models]
        self.kappa = kappa
        
    def update(self,model_mcmc):
        self.acq = [LCB(model,self.kappa) for model in model_mcmc.models]
        
    def __call__(self,x):
        x_copy = np.copy(x)
        if x.ndim == 1:
            x_copy = x_copy.reshape((1,-1))
        val_vec = [acq_model(x_copy) for acq_model in self.acq]
        val = np.mean(val_vec,axis=0)
        return val