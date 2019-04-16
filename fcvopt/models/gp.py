import numpy as np
import emcee
import warnings

from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from fcvopt.util.gp_utils import kernel_inv
from fcvopt.util.preprocess import zero_one_scale, zero_one_rescale
from fcvopt.priors.model_priors import GPPrior

class GP:
    def __init__(self,kernel,lower,upper,n_hypers=30,
                 chain_length = 10,burnin_length=150,
                 prior=None,rng=None):
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
        
        self.prior = prior
        self.X_train = None
        self.y = None
        self.burned = False
        self.eps = 1e-8
        
    def fit(self,X,y):
        
        # data
        self.X_train,self.lower,self.upper = zero_one_scale(X,self.lower,self.upper)
        if type(y) is list:
            self.y_train = np.array(y)
        else:
            self.y_train = y
        
        # kernels involved
        self.n_dim = self.X_train.shape[1]
        if self.kernel == "gaussian":
            kernel_ls= RBF(np.ones(self.n_dim))
        elif self.kernel == "matern":
            kernel_ls = Matern(np.ones(self.n_dim),nu=2.5)
            
        self.k1 = kernel_ls*C(1.0) + WhiteKernel(0.01)
          
        # initialize prior
        y_loc = np.mean(self.y_train) # mean
        y_scale = np.std(self.y_train) # std dev
        if self.prior is None:
            self.prior = GPPrior(self.n_dim,y_loc,y_scale,self.rng)
        
        # Initialize sampler
        sampler = emcee.EnsembleSampler(self.n_hypers,
                                        self.n_dim+3,
                                        self.log_posterior)
        sampler.random_state = self.rng.get_state()
        
        if not self.burned:
            self.p0 = self.prior.sample(self.n_hypers)
            
            self.p0,_,_ = sampler.run_mcmc(self.p0,self.burnin_length)
            self.burned = True
            
        pos,_,_ = sampler.run_mcmc(self.p0,self.chain_length)
        
        self.p0 = pos
        self.hypers = sampler.chain[:, -1]
        
        # 'fit' models
        self.mu_ = self.hypers[:,0]
        self.k1_ = []
        self.Kinv_ = []
        self.Kinv_y_ = []
        for theta in self.hypers:
            k1_,Kinv,Kinv_y = self._fit_gp(theta)
            self.k1_.append(k1_)
            self.Kinv_.append(Kinv)
            self.Kinv_y_.append(Kinv_y)
            
        return self
            
    def _return_kernels(self,theta):
        mu_ = theta[0]
        k1_ = self.k1.clone_with_theta(theta[1:])
        
        return mu_,k1_
        
    def log_likelihood(self,theta):
        # new kernels
        mu_,k1_ = self._return_kernels(theta)
            
        try:
            Kinv,ldet_K = kernel_inv(k1_,self.X_train,self.eps,True)
        except np.linalg.LinAlgError:
            return -np.inf
        
        y_train = np.copy(self.y_train)-mu_

        # Compute log-likelihood - not returning constant term
        log_likelihood = -0.5 * np.dot(y_train,Kinv.dot(y_train)) - 0.5*ldet_K
        
        return log_likelihood
    
    def log_posterior(self,theta):
        if np.any(theta<-15) or np.any(theta>15):
            return -np.inf
        log_lik = self.log_likelihood(theta)
        log_prior = self.prior.lnpdf(theta)
        
        if log_lik is np.NaN:
            print("Log-Likelhood compromised")
        
        if log_prior is np.NaN:
            print("Log-prior compromised")
        
        return log_lik + log_prior
    
    def _fit_gp(self,theta):
        '''
        Return quantities required for prediction down the line 
        '''
        # new kernels
        mu_,k1_ = self._return_kernels(theta)
            
        try:
            Kinv,ldet_K = kernel_inv(k1_,self.X_train,self.eps,True)
        except np.linalg.LinAlgError:
            return -np.inf
        
        y_train = np.copy(self.y_train)-mu_
        Kinv_y = Kinv.dot(y_train)
        
        return k1_,Kinv,Kinv_y
    
    def predict(self,X,scaled=False,return_std=True):
        if type(X) is not np.ndarray or X.ndim==1:
            X_copy = np.array(X).reshape(1,-1)
        else:
            X_copy = np.copy(X)
        
        if not scaled:
            X_copy,_,_ = zero_one_scale(X_copy,self.lower,self.upper)
        
        if return_std:
            y_means = []
            y_vars = np.zeros(X_copy.shape[0])
            for i in range(self.n_hypers):
                y_mean_i, y_std_i = self._predict_i(X_copy,i,return_std)
                y_means.append(y_mean_i)
                y_vars += y_std_i**2/self.n_hypers
            
            y_means = np.column_stack(y_means)
            y_mean = y_means.mean(axis=1)
            y_std = np.sqrt(y_vars + y_means.var(axis=1))
            return y_mean,y_std
        else:
            y_mean = np.column_stack([self._predict_i(X_copy,i,return_std) for i in range(self.n_hypers)]).mean(axis=1)
            return y_mean
        
    def _predict_i(self,X,i,return_std=True,return_cov=False):
        K_trans = self.k1_[i](X, self.X_train)
        y_mean = self.mu_[i] + K_trans.dot(self.Kinv_y_[i])
        
        if return_cov:
            v = self.Kinv_[i].dot(K_trans.T)
            y_cov = self.k1_[i](X) - K_trans.dot(v)  # Line 6
            return y_mean, y_cov
        elif return_std:
            # Compute variance of predictive distribution
            y_var = self.k1_[i].diag(X)
            y_var -= np.einsum("ij,ij->i",
                               np.dot(K_trans, self.Kinv_[i]), K_trans)
    
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
        
    def get_incumbent(self):
        y_mean = self.predict(self.X_train,scaled=True,return_std=False)
        inc_index = np.argmin(y_mean)
        X_inc = zero_one_rescale(self.X_train[inc_index,:].copy(),
                                 self.lower,self.upper)
        return X_inc,y_mean[inc_index]