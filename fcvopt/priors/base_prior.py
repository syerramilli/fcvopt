# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps

class UniformPrior:
    def __init__(self,lower,upper):
        self.lower = lower
        self.upper = upper
        self.d = len(lower) if type(lower) is np.ndarray else 1
        
        if  np.any(self.upper <= self.lower):
            raise Exception("Upper bounds should be greater than lower bound")
    
    def lnpdf(self,theta):
        if np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf
        else:
            return 0
        
    def sample(self,n_samples):
        return self.lower + np.random.rand(n_samples,self.d)*(self.upper-self.lower)
    
class NormalPrior:
    def __init__(self,mu=0,sigma=1):
        self.mu = mu
        self.sigma = sigma
        
    def lnpdf(self,theta):
        return np.log(sps.norm.pdf(theta,self.mu,self.sigma))
        
    def sample(self,n_samples):
        s = self.mu + self.sigma*np.random.randn(n_samples)
        s[s< -20] = -20
        return s[:,np.newaxis]
    
class HorseshoePrior:
    def __init__(self,scale=0.1):
        self.scale = scale
        
    def lnpdf(self,theta):
        return np.log(np.log(1+3.0*(self.scale/np.exp(theta))**2))
            
        
    def sample(self,n_samples):
        lam = np.random.standard_cauchy(n_samples)
        s = np.log(np.abs(self.scale*lam*np.random.randn(n_samples)))
        s[s < -20] = -20
        return s[:,np.newaxis]
    