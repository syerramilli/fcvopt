# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps

class UniformPrior:
    def __init__(self,lower,upper,rng):
        self.lower = lower
        self.upper = upper
        self.d = len(lower) if type(lower) is np.ndarray else 1
        
        if  np.any(self.upper <= self.lower):
            raise Exception("Upper bounds should be greater than lower bound")
            
        self.rng = rng
    
    def lnpdf(self,theta):
        if np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf
        else:
            return 0
        
    def sample(self,n_samples):
        return self.lower + self.rng.rand(n_samples,self.d)*(self.upper-self.lower)
    
class NormalPrior:
    def __init__(self,mu,sigma,rng):
        self.mu = mu
        self.sigma = sigma
        self.rng = rng
        
    def lnpdf(self,theta):
        return np.log(sps.norm.pdf(theta,self.mu,self.sigma))
        
    def sample(self,n_samples):
        s = self.mu + self.sigma*self.rng.randn(n_samples)
        s[s< -20] = -20
        return s[:,np.newaxis]
    
class HorseshoePrior:
    def __init__(self,scale,rng):
        self.scale = scale
        self.rng = rng
        
    def lnpdf(self,theta):
        if theta > 10:
            return -np.inf
        return np.log(np.log(1+3.0*(self.scale/np.exp(theta))**2))
            
        
    def sample(self,n_samples):
        lam = np.random.standard_cauchy(n_samples)
        s = np.log(np.abs(self.scale*lam*self.rng.randn(n_samples)))
        s[s < -20] = -20
        return s[:,np.newaxis]
    
class BetaPrior:
    def __init__(self,a,b,rng):
        self.a = a
        self.b = b
        self.rng = rng
        
    def lnpdf(self,theta):
        return sps.beta.logpdf(theta,a=self.a,b=self.b)
    
    def sample(self,n_samples):
        s = self.rng.beta(self.a,self.b,size=n_samples)
        return s[:,np.newaxis]