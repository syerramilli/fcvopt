# -*- coding: utf-8 -*-

import numpy as np
import emcee
import warnings

from copy import deepcopy
from scipy.linalg import cholesky,solve_triangular
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array
from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from fcvopt.util.preprocess import zero_one_scale, zero_one_rescale
from fcvopt.util.preprocess import standardize_vec
from fcvopt.priors.model_priors import GPPrior

class GP:
    def __init__(self,kernel,X,y,eps=1e-08):
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.d = X.shape[1]
        if kernel == "gaussian":
            self.kernel= RBF(np.ones(self.d))*C(1.0) +\
                             WhiteKernel(eps)
        elif kernel == "matern":
            self.kernel = Matern(np.ones(self.d),nu=2.5)*\
                                 C(1.0) + WhiteKernel(eps)
        self.X_train = X
        self.y_train = y                     
        self.eps = eps
    
    def fit(self, theta):
        
        if theta is None:
            self.kernel_ = clone(self.kernel)
        else:
            self.kernel_ = self.kernel.clone_with_theta(theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train)
        K[np.diag_indices_from(K)] += self.eps
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix."
                        % self.kernel_,) + exc.args
            raise
        
        L_inv = solve_triangular(L,np.eye(L.shape[0]),lower=True)
        self.K_inv = L_inv.T.dot(L_inv)
        tmp = np.ones((1,self.y_train.shape[0])).dot(self.K_inv)
        self.mu_hat = tmp.dot(self.y_train)/tmp.dot(np.ones(self.y_train.shape))
        self._comp  = self.K_inv.dot(self.y_train) - self.mu_hat*tmp.flatten()
        
        return self

    def predict(self,X,return_std=True,return_cov=False):
        """Predict using the Gaussian process regression model
        
        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.
        
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated
        
        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        
        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean
        
        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points
        
        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        
        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")
        
        
        X = check_array(X)
        
        K_trans = self.kernel_.k1(X, self.X_train)
        y_mean = self.mu_hat + K_trans.dot(self._comp)
        if return_cov:
            v = self.K_inv.dot(K_trans.T)
            y_cov = self.kernel_.k1(X) - K_trans.dot(v)  # Line 6
            return y_mean, y_cov
        elif return_std:
            # Compute variance of predictive distribution
            y_var = self.kernel_.diag(X)
            y_var -= np.einsum("ij,ij->i",
                               np.dot(K_trans, self.K_inv), K_trans)
    
            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)
        else:
            return y_mean

    def log_likelihood(self, theta=None):
        """Returns log-likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        """
        if theta is None:
            if not hasattr(self, "X_train_"):
                kernel = clone(self.kernel)
            else:
                kernel = clone(self.kernel_)
        else:
            kernel = self.kernel.clone_with_theta(theta)

        K = kernel(self.X_train)
        K[np.diag_indices_from(K)] += self.eps
        
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return -np.inf
        
        y_train = np.copy(self.y_train)
        L_inv = solve_triangular(L,np.eye(L.shape[0]))
        L_inv_y = L_inv.dot(y_train)
        L_inv_ones = L_inv.dot(np.ones(y_train.shape))
        mu_hat = L_inv_ones.dot(L_inv_y)/L_inv_ones.dot(L_inv_ones)
        y_train = y_train-mu_hat

        # Compute log-likelihood - not returning constant term
        log_likelihood = -0.5 * np.linalg.norm(L_inv_y-mu_hat*L_inv_ones)**2
        log_likelihood += -np.log(np.linalg.det(L))                                        
        
        return log_likelihood
        

class GPMCMC:
    def __init__(self,kernel,lower,upper,n_hypers=30,
                 chain_length = 10,burnin_length=50,rng=None):
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0,2e+4))
        else:
            self.rng = rng
        
        self.kernel = kernel
        self.lower = lower
        self.upper = upper
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burnin_length = burnin_length
        
        self.prior = None
        self.X_train = None
        self.y_train = None
        self.burned = False
        
    def fit(self,X,y):
        
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        
        self.X_train,self.lower,self.upper = zero_one_scale(X,self.lower,self.upper)
        self.y_train,self.y_loc,self.y_scale = standardize_vec(y)
        self.gp = GP(self.kernel,self.X_train,self.y_train)
        self.prior = GPPrior(X.shape[1],0.1,self.rng)
        
        # Initialize sampler
        sampler = emcee.EnsembleSampler(self.n_hypers,
                                        len(self.gp.kernel.theta),
                                        self.log_posterior)
        #sampler.random_state = self.rng.get_state()
        
        if not self.burned:
            self.p0 = self.prior.sample(self.n_hypers)
            
            self.p0,_,_ = sampler.run_mcmc(self.p0,self.burnin_length,
                                           rstate0=self.rng)
            self.burned = True
            
        pos,_,_ = sampler.run_mcmc(self.p0,self.chain_length,rstate0=self.rng)
        
        self.p0 = pos
        self.hypers = sampler.chain[:, -1]
        self.models = []
        
        for sampler in self.hypers:
            model = deepcopy(self.gp)
            model.fit(sampler)
            self.models.append(model)
            
    def log_posterior(self,theta):
        if np.any(theta<-20) or np.any(theta>20):
            return -np.inf
        log_lik = self.gp.log_likelihood(theta)
        log_prior = self.prior.lnpdf(theta)
        
        if log_lik is np.NaN:
            print("Log-Likelhood compromised")
        
        if log_prior is np.NaN:
            print("Log-prior compromised")
        
        return log_lik + log_prior
    
    def predict(self,X,scaled=False,return_std=True):
        if type(X) is not np.ndarray or X.ndim==1:
            X_copy = np.array(X).reshape(1,-1)
        else:
            X_copy = np.copy(X)
        
        if not scaled:
            X_copy,_,_ = zero_one_scale(X_copy,self.lower,self.upper)
        
        predictions = np.array([model.predict(X_copy) for model in self.models])
        y_mean = np.mean(predictions[:,0,:],axis=0)*self.y_scale + self.y_loc
        
        if return_std:
            y_mean_sd = np.std(predictions[:,0,:],axis=0)
            y_std = np.sqrt(np.mean(predictions[:,1,:]**2,axis=0) + y_mean_sd**2)
            return y_mean,y_std*self.y_scale
        else:
            return y_mean
        
    def get_incumbent(self):
        y_mean = self.predict(self.X_train,scaled=True,return_std=False)
        inc_index = np.argmin(y_mean)
        X_inc = zero_one_rescale(self.X_train[inc_index,:],
                                 self.lower,self.upper)
        return X_inc,y_mean[inc_index]