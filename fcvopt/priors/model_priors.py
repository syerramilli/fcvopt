# -*- coding: utf-8 -*-

import numpy as np
from fcvopt.prior.base_prior import UniformPrior, NormalPrior, HorseshoePrior

class GPPrior:
    def __init__(self,n_ls,sample_std=1,hs_scale=0.1):
        
        # length-scales
        self.n_ls = n_ls
        lower = np.log(0.1)*np.ones((n_ls,))
        upper = np.log(10)*np.ones((n_ls,))
        self.ls_prior = UniformPrior(lower,upper)
        
        # Variance:
        self.var_prior = NormalPrior(2*np.log(sample_std),1)
        
        # Noise Prior
        self.noise_prior = HorseshoePrior(hs_scale)
        
    def lnpdf(self,theta):
        log_p  = self.ls_prior.lnpdf(theta[0:self.n_ls])
        log_p += self.var_prior.lnpdf(theta[self.n_ls])
        log_p += self.noise_prior.lnpdf(theta[self.n_ls + 1])
        return log_p
    
    def sample(self,n_samples):
        ls_sample    = self.ls_prior.sample(n_samples)
        var_sample   = self.var_prior.sample(n_samples)
        noise_sample = self.noise_prior.sample(n_samples)
        return np.concatenate((ls_sample,var_sample,noise_sample),axis=1)