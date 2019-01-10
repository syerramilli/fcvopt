# -*- coding: utf-8 -*-

import numpy as np
import emcee
import warnings

from copy import deepcopy
from scipy.linalg import block_diag
from scipy.linalg import cholesky,solve_triangular,solve,det
from sklearn.base import clone
from sklearn.utils.validation import check_array
from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C


from fcvopt.util.preprocess import zero_one_scale, zero_one_rescale
from fcvopt.util.preprocess import standardize_vec
from fcvopt.priors.model_priors import AGPPrior

class AGP:
    def __init__(self,kernel,X,X_list,y,U,P,eps=1e-08):
        self.n_dim = X.shape[1]
        if kernel == "gaussian":
            kernel_ls= RBF(np.ones(self.n_dim))
        elif kernel == "matern":
            kernel_ls = Matern(np.ones(self.n_dim),nu=2.5)
            
        self.k1 = kernel_ls*C(1.0)
        self.k2 = kernel_ls*C(0.01) + C(0.0001) + WhiteKernel(0.01)
        self.X_train = X
        self.y_train = y
        self.X_list = X_list
        self.U = U
        self.P = P                 
        self.eps = eps
        
    def _kernel_inv(self,kernel,X,det=True):
        K = kernel(X)
        K[np.diag_indices_from(K)] += self.eps
        L = cholesky(K, lower=True)  # Line 2
        
        L_inv = solve_triangular(L,np.eye(L.shape[0]),lower=True)
        K_inv = L_inv.T.dot(L_inv)
        
        if det:
            ldet_K = 2*np.sum(np.log(np.diag(L)))
            return K_inv,ldet_K
        else:
            return K_inv
        
    def fit(self, theta):
        
        if theta is None:
            self.k1_ = clone(self.k1)
            self.k2_ = clone(self.k2)
        else:
            self.k1_ = self.k1.clone_with_theta(theta[np.arange(self.n_dim+1)])
            k2_params = np.append(np.arange(self.n_dim),
                                  self.n_dim+np.arange(1,4))
            self.k2_ = self.k2.clone_with_theta(theta[k2_params])
            
        # Precompute quantities required for predictions which are independent
        # of actual query points
        try:
            Sigma_n_inv = self._kernel_inv(self.k1_,self.X_train,False)
        except np.linalg.LinAlgError:
            raise
        
        n_folds = len(self.X_list)
        Ainv = [None]*n_folds
        for k in range(n_folds):
            try:
                tmp = self._kernel_inv(self.k2_,self.X_list[k],False)
            except np.linalg.LinAlgError:
                raise
            Ainv[k] = tmp
        
        Ainv = block_diag(*Ainv)
        Ainv = self.P.T.dot(Ainv).dot(self.P)
        
        tmp = self.U.T.dot(Ainv)
        inner = Sigma_n_inv + tmp.dot(self.U)
        tmp2 = solve(inner,tmp)
        self.K_inv = Ainv - tmp.T.dot(tmp2)
        
        tmp = np.ones((1,self.y_train.shape[0])).dot(self.K_inv)
        self.mu_hat = (tmp.dot(self.y_train)/tmp.dot(np.ones(self.y_train.shape)))[0]
        self._comp  = self.K_inv.dot(self.y_train) \
                                    - self.mu_hat*tmp.flatten()

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
        
        K_trans = self.k1_(X, self.X_train)
        K_trans =  K_trans.dot(self.U.T)
        y_mean = self.mu_hat + K_trans.dot(self._comp)
        if return_cov:
            v = self.K_inv.dot(K_trans.T)
            y_cov = self.k1_(X) - K_trans.dot(v)  # Line 6
            return y_mean, y_cov
        elif return_std:
            # Compute variance of predictive distribution
            y_var = self.k1_.diag(X)
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
            k1_ = clone(self.k1)
            k2_ = clone(self.k2)
        else:
            k1_ = self.k1.clone_with_theta(theta[np.arange(self.n_dim+1)])
            k2_params = np.append(np.arange(self.n_dim),
                                  self.n_dim+np.arange(1,4))
            k2_ = self.k2.clone_with_theta(theta[k2_params])
            
        # Precompute quantities required for predictions which are independent
        # of actual query points
        try:
            Sigma_n_inv,ldet_K = self._kernel_inv(k1_,self.X_train,True)
        except np.linalg.LinAlgError:
            return -np.inf
        
        n_folds = len(self.X_list)
        Ainv = [None]*n_folds
        for k in range(n_folds):
            try:
                tmp,tmp2 = self._kernel_inv(k2_,self.X_list[k],True)
            except np.linalg.LinAlgError:
                return -np.inf
            Ainv[k] = tmp
            ldet_K += tmp2
        
        Ainv = block_diag(*Ainv)
        Ainv = self.P.T.dot(Ainv).dot(self.P)
        
        tmp = self.U.T.dot(Ainv)
        inner = Sigma_n_inv + tmp.dot(self.U)
        tmp2 = solve(inner,tmp)
        ldet_K += np.log(det(inner))
        K_inv = Ainv - tmp.T.dot(tmp2)
        
        y_train = np.copy(self.y_train)
        tmp = np.ones((1,y_train.shape[0])).dot(K_inv)
        mu_hat = (tmp.dot(y_train)/tmp.dot(np.ones(y_train.shape)))[0]
        y_train = y_train-mu_hat

        # Compute log-likelihood - not returning constant term
        log_likelihood = -0.5 * np.dot(y_train,K_inv.dot(y_train)) - 0.5*ldet_K
        
        return log_likelihood

class AGPMCMC:
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
        self.y_list = None
        self.f_list = None
        self.burned = False
        
    def fit(self,X,y_list,f_list):
        
        # data
        self.X_train,self.lower,self.upper = zero_one_scale(X,self.lower,self.upper)
        y = np.array([subitem for item in y_list for subitem in item])
        self.y_train,self.y_loc,self.y_scale = standardize_vec(y)
        self.f_list = f_list
        
        # create permuation matrix
        f_full = np.array([subitem for item in f_list for subitem in item])
        self.N = f_full.size
        self.groups = np.unique(f_full)
        P = np.zeros((self.N,self.N))
        P[np.arange(self.N),np.argsort(f_full)] = 1
        
        # partition points by fold/group
        n_reps = [len(f_vec) for f_vec in f_list]
        U = block_diag(*[np.ones((n_rep,1)) for n_rep in n_reps])
        
        X_aug = np.repeat(self.X_train,n_reps,axis=0)
        n_dim = len(self.lower)
        X_list = [X_aug[np.where(f_full==group)[0],0:n_dim] for group in self.groups] 
          
        # initialize GP
        self.gp = AGP(self.kernel,self.X_train,X_list,self.y_train,U,P)
        self.prior = AGPPrior(X.shape[1],0.1,self.rng)
        
        # Initialize sampler
        sampler = emcee.EnsembleSampler(self.n_hypers,
                                        n_dim+4,
                                        self.log_posterior)
        sampler.random_state = self.rng.get_state()
        
        if not self.burned:
            self.p0 = self.prior.sample(self.n_hypers)
            
            self.p0,_,_ = sampler.run_mcmc(self.p0,self.burnin_length)
            self.burned = True
            
        pos,_,_ = sampler.run_mcmc(self.p0,self.chain_length)
        
        self.p0 = pos
        self.hypers = sampler.chain[:, -1]
        self.models = []
        
        for sampler in self.hypers:
            model = deepcopy(self.gp)
            model.fit(sampler)
            self.models.append(model)
            
    def log_posterior(self,theta):
        if np.any(theta<-15) or np.any(theta>15):
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
    
    def _fold_var(self,x,fold_ids):
        x_copy = np.array(x).reshape(1,-1)
        x_copy,_,_ = zero_one_scale(x_copy,self.lower,self.upper)
        
        msp_vec = np.zeros(len(self.models))
        msp = []
        for fold_id in fold_ids:   
            for i,m in enumerate(self.models):
                k1_x = m.k1_(x_copy,m.X_train).dot(m.U.T)
    
                k2_x = np.hstack([m.k2_(x_copy,m.X_list[i]) if fold_id == group \
                                  else np.zeros((1,m.X_list[i].shape[0])) \
                                  for i,group in enumerate(self.groups)])
                
                r = k1_x + k2_x.dot(m.P.T)
                tmp = r.dot(m.K_inv)
                k1_var = m.k1_(x_copy)
                total_var = k1_var + m.k2_(x_copy)
                den = total_var - np.inner(r,tmp)
                
                term1 = m.K_inv + np.outer(tmp,tmp)/den
                
                msp_vec[i] = k1_var- k1_x.dot(term1).dot(k1_x.T) +\
                              2*k1_var*tmp.dot(k1_x.T)/den -\
                                     k1_var**2/den
            msp.append(np.mean(msp_vec[i]))
                                      
        return msp

        
        
        
        
        
