# -*- coding: utf-8 -*-

import numpy as np

class ImprovLCB:
    def __init__(self,model,x_inc,kappa=2):
        self.model = model
        self.x_inc = x_inc
        self.kappa = kappa
        
    def update(self,model,x_inc):
        self.model = model
        self.x_inc = x_inc
        
    def __call__(self,x):        
        X = np.row_stack((x,self.x_inc))
        y_mean,y_cov = self.model.predict(X,return_std=False,return_cov=True)
        tmp = np.max(np.trace(y_cov)-2*y_cov[0,1],0)
        val = y_mean[0]-y_mean[1] -self.kappa*np.sqrt(tmp)
        
        den = 1
        if hasattr(self.model,"kernel_"):
            # Regular GP
            den = 2*np.exp(0.5*self.model.kernel_.k1.theta[-1])
        elif hasattr(self.model,"k1_"):
            # AGP
            den = 2*np.exp(0.5*self.model.k1_.theta[-1])
        return val/den
    

class ImprovLCBMCMC:
    def __init__(self,model_mcmc,x_inc,kappa=2):
        self.acq = [ImprovLCB(model,x_inc,kappa) for model in model_mcmc.models]
        self.den = np.sqrt(np.sum([np.exp(model.kernel_.k1.theta[-1]) for model in model_mcmc.models]))
        self.kappa = kappa
        
    def update(self,model_mcmc,x_inc):
        self.acq = [ImprovLCB(model,x_inc,self.kappa) for model in model_mcmc.models]
        self.den = np.sqrt(np.sum([np.exp(model.kernel_.k1.theta[-1]) for model in model_mcmc.models]))
        
    def __call__(self,x):
        val_vec = [acq_model(x) for acq_model in self.acq]
        val = np.mean(val_vec,axis=0)
        return val#/self.den