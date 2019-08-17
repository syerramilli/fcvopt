import numpy as np
import emcee
import warnings

from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from fcvopt.models.basemodel import BaseModel
from fcvopt.util.gp_utils import kernel_inv
from fcvopt.util.preprocess import zero_one_scale, zero_one_rescale
from fcvopt.priors.model_priors import GPPrior

class GP(BaseModel):
    """
    Gaussian process model that uses MCMC sampling to marginalize over
    the hyper-parameters. 
    
    Parameters
    -------------
    kernel: string
        Name of the kernel. Current options are "gaussian" (Gaussian or 
        RBF Kernel) and "matern" (Matern 5/2 kernel). These kernels are 
        implemented in scikit-learn
        
    lower: array-like, shape = (n_dim,)
        Lower bound on the inpute space used for scaling to the 0-1 hypercube
    
    upper: array-like, shape = (n_dim,)
        Upper bound on the features used for scaling to the 0-1 hypercube
    
    n_hypers: int, optional (default:30)
         The number of hyperparameter samples. This is also the determine
         the number of walkers for MCMC sampling. 
        
    chain_length: int, optional (default:100)
        The number of MCMC steps. The walkers in the last step will be 
        used as the hyperparameter samples.
    
    burnin_length: int, optional (default:150)
        The number of burnin steps before the actual MCMC sampling begins
    
    prior: object, optional (default: None)
        Prior on the GP hyper-parametes. Uses GPPrior class if nothing
        is specifed. Check the documentation on GPPrior for the 
        required methods.
        
    rng: int, RandomState instance or None, optional (default: None)
        The generators used to intialize the MCMC samples. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    
    Attributes
    -------------
    X: array, shape = (n_samples,n_dim):
        Feature values in training data (scaled to the 0-1 hypercube)
        
    y : array, shape = (n_samples,)
        Target values in training data 
        
    k1: kernel object 
        The kernel with the specified structure (not used for prediction)
        
    prior:
        The prior on the GP hyperparameters
        
    burned: bool
        Indicates whether burning has been performed. False before 
        the fit method is used
    
    mu_: array, shape = (n_hypers,)
        Samples from the posterior distribution on the GP mean
        
    hypers: array, shape = (n_hypers,n_features+3)
        Hyperparameter samples from the posterior. Includes the mean,
        length-scales, amplitude, and the noise variance. All 
        hyperparameters except the mean are in log-scale
        
    k1_: list, length = n_hypers
        List of kernel objects, each of which correspond to the samples
        in hypers
        
    Kinv_: list, length = n_hypers
        List of inverse covariance kernel in ``.X`` corresponding to
        the samples in hypers. Shape of each array is (n_samples,n_samples)
        
    Kinv_y_: list, length = n_hypers
        List of dot-products of the invariance covariance kernels
        corresponding to the samples in hypers and `.y`. 
        (Used for prediction)
        
    p0: array, shape = (n_hypers,n_features+3)
        Last-state of the MCMC chain
    
    """
    def __init__(self,kernel,lower,upper,n_hypers=30,
                 chain_length = 100,burnin_length=100,
                 prior=None,rng=None):
        super(GP,self).__init__()
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0,2e+4))
        elif type(rng) is int:
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = rng
        
        self.kernel = kernel
        self.lower = lower
        self.upper = upper
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burnin_length = burnin_length
        self.prior = prior
        self.burned = False
        self.eps = 1e-8

        # quantity not needed for GP but initialized for inheritance later
        self.U = None
        
    def fit(self,X,y):
        '''
        Samples hyperparameters from the posterior distribution using
        MCMC sampling and then trains a GP on the data for each sample
        
        Parameters
        -----------
        X: array, shape = (n_samples,n_dim)
            Input data. 
        
        y: array, shape = (n_samples,)
            Corresponding target values
        
        '''
        # data - appropriate transformations
        self.X,self.lower,self.upper = zero_one_scale(X,self.lower,self.upper)
        if type(y) is list:
            self.y = np.array(y)
        else:
            self.y = y
        
        # define kernel structure 
        kernel_ls = self._initialize_kernel_ls()
        self.k1 = kernel_ls*C(1.0) + WhiteKernel(0.01)
          
        # initialize prior
        y_min = np.min(self.y) # mean
        y_max = np.max(self.y) # mean
        y_scale = np.std(self.y) # std dev
        if self.prior is None:
            self.prior = GPPrior(self.n_dim,y_min,y_max,y_scale,self.rng)
        
        # Initialize sampler
        sampler = emcee.EnsembleSampler(self.n_hypers,
                                        self.n_dim+3,
                                        self.log_posterior)
        sampler.random_state = self.rng.get_state()
        
        # Perform MCMC sampling
        if not self.burned:
            self.p0 = self.prior.sample(self.n_hypers)
            
            self.p0,_,_ = sampler.run_mcmc(self.p0,self.burnin_length)
            self.burned = True
            
        pos,_,_ = sampler.run_mcmc(self.p0,self.chain_length)
        
        self.p0 = pos
        self.hypers = sampler.chain[:, -1]
        
        # 'fit' GP models for each hyperparameter sample
        self.mu_ = self.hypers[:,0]
        self.k1_ = []
        self.Kinv_ = []
        self.Kinv_y_ = []
        for theta in self.hypers:
            k1_,Kinv,Kinv_y = self._fit_gp(theta)
            self.k1_.append(k1_)
            self.Kinv_.append(Kinv)
            self.Kinv_y_.append(Kinv_y)
        
        self.is_trained = True
        return self
        
    def _initialize_kernel_ls(self):
        self.n_dim = self.X.shape[1]
        if self.kernel == "gaussian":
            kernel_ls= RBF(np.ones(self.n_dim))
        elif self.kernel == "matern":
            kernel_ls = Matern(np.ones(self.n_dim),nu=2.5)
        return kernel_ls


    def _return_kernels(self,theta):
        mu_ = theta[0]
        k1_ = self.k1.clone_with_theta(theta[1:])
        
        return mu_,k1_
        
    def log_likelihood(self,theta):
        '''
        Returns the unnormalized log-likelhood of theta for training data
        
        Parameters
        -----------
        theta: array, shape = (n_dim+3,)
            Kernel hyperparameters for which the log-likelihood is computed
            
        Returns
        ----------
        log_posterior: float
            Unnormalized log-likelihood of theta for training data
        '''
        # new kernels
        mu_,k1_ = self._return_kernels(theta)
            
        try:
            Kinv,ldet_K = kernel_inv(k1_,self.X,self.eps,True)
        except np.linalg.LinAlgError:
            return -np.inf
        
        y_train = np.copy(self.y)-mu_

        # Compute log-likelihood - not returning constant term
        log_likelihood = -0.5 * np.dot(y_train,Kinv.dot(y_train)) - 0.5*ldet_K
        
        return log_likelihood
    
    def log_posterior(self,theta):
        '''
        Returns the unnormalized log-posterior of theta for training data
        
        Parameters
        -----------
        theta: array, shape = (n_dim+3,)
            Kernel hyperparameters for which the (unnormalized) 
            log-posterior is computed
            
        Returns
        ----------
        log_posterior: float
            Unnormalized log-posterior of theta for training data
        '''
        if np.any(theta<-15) or np.any(theta>15):
            return -np.inf
        
        log_prior = self.prior.lnpdf(theta)
        
        if np.isinf(log_prior):
            return -np.inf
        
        log_lik = self.log_likelihood(theta)
        
        if np.isnan(log_lik):
            print(theta)
            print("Log-Likelhood compromised")
        
        return log_lik + log_prior
    
    def _fit_gp(self,theta):
        '''
        Return quantities required for prediction down the line 
        '''
        # new kernels
        mu_,k1_ = self._return_kernels(theta)
            
        try:
            Kinv,_ = kernel_inv(k1_,self.X,self.eps,False)
        except np.linalg.LinAlgError:
            return -np.inf
        
        y_train = np.copy(self.y)-mu_
        Kinv_y = Kinv.dot(y_train)
        
        return k1_,Kinv,Kinv_y
    
    def predict(self,X,scaled=False,return_std=True):
        """
        Returns the predictive mean and variance at the given points. 
        Requires model to be fit.
        
        Parameters
        -----------
        X: array-like, shape = (N,n_dim)
            Query points
        scaled: bool, optional (default:False)
            Flag indicating whether the supplied points have been 
            scaled to the 0-1 hypercube
        return_std: bool, optional (default:True)
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        
        Returns
        ----------
        y_mean: array, shape = (N,)
            Mean of the predictive distribution at the query points
        
        y_std: array, shape = (N,) optional
            Standard deviation of the predictive distribution at the query 
            points. Returned only when return_std = True.
        """
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
        K_trans = self.k1_[i](X, self.X)
        if self.U is not None:
            # while not needed here, this is needed for
            # the AGP class which derives from it
            K_trans = K_trans * self.U.T 
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
        """
        Returns the observed minima and its predicted value
        
        Returns
        ----------
        x_inc: array (n_dim,):
            observed minima
        y_inc: float
            predicted value at the minima
        """
        y_mean = self.predict(self.X,scaled=True,return_std=False)
        inc_index = np.argmin(y_mean)
        X_inc = zero_one_rescale(self.X[inc_index,:].copy(),
                                 self.lower,self.upper)
        return X_inc,y_mean[inc_index]