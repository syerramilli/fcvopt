# -*- coding: utf-8 -*-

import numpy as np
from fcvopt.priors.base_prior import UniformPrior, NormalPrior
from fcvopt.priors.base_prior import HorseshoePrior, BetaPrior

class GPPrior:
    def __init__(self,n_ls,hs_scale,rng):
        
        # length-scales
        self.n_ls = n_ls
        lower = np.log(0.01)*np.ones((n_ls,))
        upper = np.log(10)*np.ones((n_ls,))
        self.ls_prior = UniformPrior(lower,upper,rng)
        
        # Variance:
        self.var_prior = NormalPrior(0,1,rng)
        
        # Noise Prior
        self.noise_prior = HorseshoePrior(hs_scale,rng)
        
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
    
    
class AGPPrior:
    def __init__(self,n_ls,mean,amp,rng):
        # mean-prior:
        self.mean_prior = NormalPrior(mean,amp,rng)
        
        # length-scales
        self.n_ls = n_ls
        lower = np.log(0.05)*np.ones((n_ls,))
        upper = np.log(10)*np.ones((n_ls,))
        self.ls_prior = UniformPrior(lower,upper,rng)
        
        # variance terms
        self.var_prior = NormalPrior(2*np.log(amp),1,rng)
        self.rho_prior = UniformPrior(0.7,1,rng)
        self.rho2_prior = UniformPrior(0,1,rng)
        
        # noise prior
        self.noise_prior = HorseshoePrior(amp,rng)
        
    def lnpdf(self,theta):
        # mean
        log_p = self.mean_prior.lnpdf(theta[0])
        
        # length scales
        log_p  += self.ls_prior.lnpdf(theta[1:(self.n_ls+1)])
        
        # variances - first calulate base quantities
        dev_var = np.sum(np.exp(theta[self.n_ls+2+np.arange(2)]))
        total_var = np.exp(theta[self.n_ls+1]) + dev_var
        rho = 1-dev_var/total_var
        rho2 = np.exp(theta[self.n_ls+2])/dev_var
        total_var = np.log(total_var)
        
        log_p += self.var_prior.lnpdf(total_var) + self.rho_prior.lnpdf(rho)
        log_p += self.rho2_prior.lnpdf(rho2)
        log_p += - 2*total_var - np.log(1-rho) 
        log_p += np.sum(theta[self.n_ls+1+np.arange(3)])
        
        log_p += self.noise_prior.lnpdf(theta[self.n_ls + 4])
        return log_p
    
    def sample(self,n_samples):
        mean_sample = self.mean_prior.sample(n_samples)
        
        ls_sample = self.ls_prior.sample(n_samples)
        
        total_var_sample   = self.var_prior.sample(n_samples)
        rho_sample = self.rho_prior.sample(n_samples)
        rho2_sample = self.rho2_prior.sample(n_samples)
        
        sigf_sample = np.log(rho_sample) + total_var_sample
        sigd1_sample = np.log(rho2_sample * (1-rho_sample)) + total_var_sample
        sigd2_sample = np.log((1-rho2_sample) * (1-rho_sample)) + total_var_sample
        
        noise_sample = self.noise_prior.sample(n_samples)
        return np.concatenate((mean_sample,ls_sample,sigf_sample,
                               sigd1_sample,sigd2_sample,
                               noise_sample),axis=1)