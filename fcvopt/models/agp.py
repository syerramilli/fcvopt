import numpy as np
import emcee
import warnings

from scipy.sparse import block_diag
from scipy.linalg import cholesky,cho_solve
from sklearn.gaussian_process.kernels import RBF, Matern,WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from fcvopt.util.gp_utils import kernel_inv
from fcvopt.util.preprocess import zero_one_scale, zero_one_rescale
from fcvopt.priors.model_priors import AGPPrior

class AGP:
    """
    Additive Gaussian process model that uses MCMC sampling to marginalize over
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
        
    rng: int, RandomState instance or None, optional (default: None)
        The generators used to intialize the MCMC samples. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    
    Attributes
    -------------
    X_train: array, shape = (n_unique,n_dim):
        Feature values in training data (scaled to the 0-1 hypercube)
        
    y_list: list shape = (n_samples,)
        Target values in training data
        
    f_list: list
        Fold identities
        
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
        List of inverse covariance kernel in ``X_train`` corresponding to
        the samples in hypers. Shape of each array is (n_samples,n_samples)
        
    Kinv_y_: list, length = n_hypers
        List of dot-products of the invariance covariance kernels
        corresponding to the samples in hypers and `y_train`. 
        (Used for prediction)
        
    p0: array, shape = (n_hypers,n_features+3)
        Last-state of the MCMC chain
    
    """
    def __init__(self,kernel,lower,upper,n_hypers=30,
                 chain_length = 100,burnin_length=100,rng=None):
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
        self.eps = 1e-8
        
    def fit(self,X,y_list,f_list):
        
        # data
        self.X_train,self.lower,self.upper = zero_one_scale(X,self.lower,self.upper)
        self.y_train = np.array([subitem for item in y_list for subitem in item])
        self.f_list = f_list
        
        # kernels involved
        self.n_dim = self.X_train.shape[1]
        if self.kernel == "gaussian":
            kernel_ls= RBF(np.ones(self.n_dim))
        elif self.kernel == "matern":
            kernel_ls = Matern(np.ones(self.n_dim),nu=2.5)
            
        self.k1 = kernel_ls*C(1.0)
        self.k2 = kernel_ls*C(0.01) + C(0.0001) + WhiteKernel(0.01)
        
        # create permuation matrix
        f_full = np.array([subitem for item in f_list for subitem in item])
        self.N = f_full.size
        self.groups = np.unique(f_full)
        #self.P = np.zeros((self.N,self.N))
        #self.P[np.arange(self.N),np.argsort(f_full)] = 1
        # more economical calculations
        self.P = np.argsort(f_full)
        self.PT = np.argsort(self.P)
        
        # partition points by fold/group
        n_reps = [len(f_vec) for f_vec in f_list]
        self.U = block_diag([np.ones((n_rep,1)) for n_rep in n_reps])
        
        X_aug = np.repeat(self.X_train,n_reps,axis=0)
        self.X_list = [X_aug[np.where(f_full==group)[0],0:self.n_dim] for group in self.groups]
          
        # initialize prior
        y_min = np.min(self.y_train) # minimum
        y_max = np.max(self.y_train) # maximum
        y_scale = np.std(self.y_train) # std dev
        self.prior = AGPPrior(self.n_dim,y_min,y_max,y_scale,self.rng)
        
        # Initialize sampler
        sampler = emcee.EnsembleSampler(self.n_hypers,
                                        self.n_dim+5,
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
        self.k2_ = []
        self.Kinv_ = []
        self.Kinv_y_ = []
        for theta in self.hypers:
            k1_,k2_,Kinv,Kinv_y = self._fit_gp(theta)
            self.k1_.append(k1_)
            self.k2_.append(k2_)
            self.Kinv_.append(Kinv)
            self.Kinv_y_.append(Kinv_y)
            
        return self
            
    def _return_kernels(self,theta):
        mu_ = theta[0]
        k1_ = self.k1.clone_with_theta(theta[np.arange(1,self.n_dim+2)])
        k2_params = theta[np.append(1+np.arange(self.n_dim),
                                    self.n_dim+np.arange(2,5))]
        k2_ = self.k2.clone_with_theta(k2_params)
        
        return mu_,k1_,k2_
        
    def log_likelihood(self,theta):
        # new kernels
        mu_,k1_,k2_ = self._return_kernels(theta)
            
        # Precompute quantities required for predictions which are independent
        # of actual query points
        try:
            Sigma_n_inv,ldet_K = kernel_inv(k1_,self.X_train,self.eps,True)
        except np.linalg.LinAlgError:
            return -np.inf
        
        n_folds = len(self.X_list)
        Ainv = np.zeros((self.N,self.N))
        p = 0
        for k in range(n_folds):
            try:
                tmp,tmp2 = kernel_inv(k2_,self.X_list[k],self.eps,True)
            except np.linalg.LinAlgError:
                return -np.inf
            n = self.X_list[k].shape[0]
            ind = p + np.arange(n)
            Ainv[ind[:,None],ind] = tmp
            p += n
            ldet_K += tmp2
        
        #Ainv = self.P.T.dot(Ainv).dot(self.P)
        # implementing permutation P^T.Ainv.P
        for l in range(self.N):
            Ainv[:,l] = Ainv[self.PT,l]
        for l in range(self.N):
            Ainv[l,:] = Ainv[l,self.PT]         
        
        tmp = self.U.T.dot(Ainv)
        inner = Sigma_n_inv + tmp * self.U
        inner_chol = cholesky(inner,check_finite=False)
        tmp2 = cho_solve((inner_chol,False),tmp)
        ldet_K += 2*np.sum(np.log(np.diag(inner_chol)))
        K_inv = Ainv - tmp.T.dot(tmp2)
        
        y_train = np.copy(self.y_train)-mu_

        # Compute log-likelihood - not returning constant term
        log_likelihood = -0.5 * np.dot(y_train,K_inv.dot(y_train)) - 0.5*ldet_K
        
        return log_likelihood
    
    def log_posterior(self,theta):
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
        mu_,k1_,k2_ = self._return_kernels(theta)
            
        # Precompute quantities required for predictions which are independent
        # of actual query points
        try:
            Sigma_n_inv = kernel_inv(k1_,self.X_train,self.eps,False)
        except np.linalg.LinAlgError:
            return -np.inf
        
        n_folds = len(self.X_list)
        Ainv = np.zeros((self.N,self.N))
        p = 0
        for k in range(n_folds):
            try:
                tmp = kernel_inv(k2_,self.X_list[k],self.eps,False)
            except np.linalg.LinAlgError:
                return -np.inf
            n = self.X_list[k].shape[0]
            ind = p + np.arange(n)
            Ainv[ind[:,None],ind] = tmp
            p += n
            
        #Ainv = self.P.T.dot(Ainv).dot(self.P)
        # implementing permutation P^T.Ainv.P
        for l in range(self.N):
            Ainv[:,l] = Ainv[self.PT,l]
        for l in range(self.N):
            Ainv[l,:] = Ainv[l,self.PT]
        
        tmp = self.U.T.dot(Ainv)
        inner = Sigma_n_inv + tmp * self.U
        inner_chol = cholesky(inner,check_finite=False)
        tmp2 = cho_solve((inner_chol,False),tmp)
        K_inv = Ainv - tmp.T.dot(tmp2)
        
        y_train = np.copy(self.y_train)-mu_
        Kinv_y = K_inv.dot(y_train)
        
        return k1_,k2_,K_inv,Kinv_y
    
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
        y_mean = self.predict(self.X_train,scaled=True,return_std=False)
        inc_index = np.argmin(y_mean)
        X_inc = zero_one_rescale(self.X_train[inc_index,:].copy(),
                                 self.lower,self.upper)
        return X_inc,y_mean[inc_index], inc_index
    
    def _fold_var(self,x,fold_ids):
        x_copy = np.array(x).reshape(1,-1)
        x_copy,_,_ = zero_one_scale(x_copy,self.lower,self.upper)
        
        msp_vec = np.zeros(self.n_hypers)
        msp = []
        for fold_id in fold_ids:   
            for i in range(self.n_hypers):
                k1_x = self.k1_[i](x_copy,self.X_train) * self.U.T
    
                k2_x = np.hstack([self.k2_[i](x_copy,self.X_list[i]) if fold_id == group \
                                  else np.zeros((1,self.X_list[i].shape[0])) \
                                  for i,group in enumerate(self.groups)])
                
                #r = k1_x + k2_x.dot(self.P.T)
                r = k1_x + k2_x[0:1,self.P]
                tmp = r.dot(self.Kinv_[i])
                k1_var = self.k1_[i](x_copy)
                total_var = k1_var + self.k2_[i](x_copy)
                den = total_var - np.inner(r,tmp)
                
                term1 = self.Kinv_[i] + np.outer(tmp,tmp)/den
                
                msp_vec[i] = k1_var- k1_x.dot(term1).dot(k1_x.T) +\
                              2*k1_var*tmp.dot(k1_x.T)/den -\
                                     k1_var**2/den
            msp.append(np.mean(msp_vec))
                                      
        return msp