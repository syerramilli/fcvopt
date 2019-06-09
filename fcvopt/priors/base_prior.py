# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps

class UniformPrior:
    """
    Uniform prior
    
    Can be defined for single or multiple variables. In the case of 
    multiple variables, the distributions are independent, each with their 
    respective lower and upper bounds
    
    Parameters
    ----------------
    lower: 1-D array or float 
        Lower bound of the prior. For a single variable, supply flower.
        Else, supply a 1-D array whose length is equal to the number
        of variables
        
    upper: 1-D array or float
        Upper bound of the prior. For a single variable, supply flower.
        Else, supply a 1-D array whose length is equal to the number
        of variables
        
    rng: np.random.RandomState instance
        The random number generator
        
    Attributes
    ----------------
    lower, upper, rng: as defined in the arguments
    
    d: int
        The dimension of the random-vector
    """
    def __init__(self,lower,upper,rng):
        self.lower = lower
        self.upper = upper
        self.d = len(lower) if type(lower) is np.ndarray else 1
        
        if  np.any(self.upper <= self.lower):
            raise Exception("Upper bounds should be greater than lower bound")
            
        self.rng = rng
    
    def lnpdf(self,theta):
        """
        Returns the (un-normalized) log-probability of theta.
        
        Parameters
        -------------
        theta: 1-D array or float
            For a single variable, supply float. 
            
        Returns
        -------------
        log_density: float or -np.inf
            Returns 0 if all variables are within their respective bounds.
            Else returns `-np.inf`.
        """
        if np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf
        else:
            return 0
        
    def sample(self,n_samples):
        """
        Sample from the prior
        
        Parameters
        -------------
        n_samples: int
            The number of samples to be returned
            
        Returns
        -------------
        theta_matrix: 2-d array of shape (n_samples,d)
            Samples from the prior distribution with the columns 
            corresponding to the different random-variables
        """
        return self.lower + self.rng.rand(n_samples,self.d)*(self.upper-self.lower)
    
class NormalPrior:
    """
    Normal prior
    
    Can be defined only for a single random variable
    
    Parameters
    ---------------
    mu: float
        The mean of the distribution
        
    sigma: float
        The standard deviation of the distribution
        
    rng: np.random.RandomState instance
        The random number generator
    """
    def __init__(self,mu,sigma,rng):
        self.mu = mu
        self.sigma = sigma
        self.rng = rng
        
    def lnpdf(self,theta):
        """
        Returns the (normalized) log-probability of theta.
        
        Parameters
        -------------
        theta: float. 
            
        Returns
        -------------
        log_density: float
        """
        return np.log(sps.norm.pdf(theta,self.mu,self.sigma))
        
    def sample(self,n_samples):
        """
        Sample from the prior
        
        Parameters
        -------------
        n_samples: int
            The number of samples to be returned
            
        Returns
        -------------
        theta_matrix: 2-d array of shape (n_samples,1)
            Samples from the prior distribution
        """
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
        """
        Sample from the prior
        
        Parameters
        -------------
        n_samples: int
            The number of samples to be returned
            
        Returns
        -------------
        theta_matrix: 2-d array of shape (n_samples,1)
            Samples from the prior distribution
        """
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