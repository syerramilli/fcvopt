import numpy as np
import time
import os
import pickle
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from fcvopt.models.gp import GP
from fcvopt.acquisition import LCB
from fcvopt.util.samplers import lh_sampler
from fcvopt.util.wrappers import scipy_minimize 
from fcvopt.util.preprocess import zero_one_scale


class BayesOpt:
    def __init__(self,estimator,param_bounds,metric,n_folds=10,logscale=None,
                 integer = [],return_prob=False,kernel="matern",kappa=2,
                 n_init=4,max_iter=10,verbose=0,seed=None,save_iter=None,
                 save_dir=None):
        self.estimator = estimator
        self.param_names = list(param_bounds.keys())
        self.param_bounds = np.array(list(param_bounds.values()),
                                     dtype = np.float
                                    )
        if logscale is not None:
            self.param_bounds[logscale,:] = np.log(self.param_bounds[logscale,:])
        self.logscale = logscale
        self.integer = integer
        self.metric = metric
        self.return_prob = return_prob
        
        self.rng = np.random.RandomState(seed=seed)
        self.cv = KFold(n_splits=n_folds,shuffle=True,random_state=self.rng)
        self.n_folds = n_folds
        
        self.kernel = kernel
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.kappa = kappa
        
        # if saving, then
        self.save_iter = save_iter
        self.save_dir = save_dir
        
        self.gp = None
        self.X = None
        self.y = []
        self.eval_time = []
        self.total_time = None
        self.folds = None

        # Not needed in regular Bayesian optimization
        # but needed in FCVOpt
        self.f_list = None
        
    def run(self,X_alg,y_alg,fold_index=None):
        
        start_time = time.time()
        if self.gp is None:
            n_dim = self.param_bounds.shape[0]
            if type(self.n_init) is int:
                self.X = lh_sampler(self.n_init,self.param_bounds[:,0],
                                    self.param_bounds[:,1],self.rng)
            else:
                self.X = self.n_init
                self.n_init = self.X.shape[0]
                        
            self.folds = [ind for ind in self.cv.split(X_alg)]
            # evaluate all folds
            if fold_index is None:
                self.fold_index = np.arange(self.n_folds)
            else:
                self.fold_index = [fold_index]
                
            for i in np.arange(self.n_init):
                tmp1,tmp2 = self._fold_eval(self.X[i,:],self.fold_index,
                                            X_alg,y_alg,True)
                self.y.append(tmp1)
                self.eval_time.append(tmp2)
                
            n_hypers = (n_dim+5)*3
            if n_hypers % 2 == 1:
                n_hypers +=1
                
            self.gp = GP(self.kernel,self.param_bounds[:,0],
                         self.param_bounds[:,1],
                         n_hypers=n_hypers,
                         chain_length=100,rng=self.rng,
                         burnin_length=100)
            self.acq = None
            
            self.X_inc = np.zeros((self.max_iter,n_dim))
            self.y_inc = np.zeros((self.max_iter,))
            self.acq_vec = np.zeros((self.max_iter,))
            self.sigma_f_vec = np.zeros((self.max_iter,))
            
            # gp timers
            self.mcmc_time = np.zeros((self.max_iter,))
            self.acq_time = np.zeros((self.max_iter,))
            
        output_header = '%6s %9s %9s %10s' % \
                    ('iter', 'f_best', 'acq_best',"sigma_f")
        
        for i in range(self.max_iter):
            # Fit GP model and get incumbent
            x_inc = self._fitgp_and_inc(i)
            
            # acquisition function optimization - find candidate
            x_cand, acq_cand = self._acquistion(x_inc,i)
            
            if self.verbose >= 2:
                if i%10==0:
                    # print header every 10 iterations
                    print(output_header)
                print('%6i %3.3e %3.3e %3.3e' %\
                      (i, self.y_inc[i],acq_cand,self.sigma_f_vec[i]))
                
            self.acq_vec[i] = acq_cand
            
            if self.save_iter is not None:
                if (i+1)%self.save_iter == 0:
                    filepath = os.path.join(self.save_dir,"iter_"+str(i)+".pkl")
                    self.save_to_pickle(filepath)
                        
            if i < self.max_iter-1:
                
                # evaluate candidate
                y_cand,time_cand = self._fold_eval(x_cand,self.fold_index,
                                                   X_alg,y_alg,True)
                
                # append observations
                self.X = np.vstack((self.X,np.copy(x_cand)))
                self.y.append(y_cand)
                self.eval_time.append(time_cand)
        
        self.total_time = time.time()-start_time
            
        # Final output message
        return self.print_and_return(x_cand)
        
    def term_crit(self):
        return (self.y_inc-self.acq_vec)/self.sigma_f_vec/self.acq.kappa
    
    def _fitgp_and_inc(self,i):
        mcmc_start = time.time()
        if self.f_list is not None:
            self.gp.fit(self.X,self.y,self.f_list)
        else:
            self.gp.fit(self.X,self.y)
        self.mcmc_time[i] = time.time()-mcmc_start
        
        
        self.sigma_f_vec[i] = \
                        np.sqrt(np.mean([np.exp(self.gp.k1_[j].theta[self.gp.n_dim]) \
                                        for j in range(self.gp.n_hypers)]) + \
                                np.var(self.gp.mu_))
        
        ########## Find incumbent ##########
        self.X_inc[i,:],self.y_inc[i] = self.gp.get_incumbent()
        
        # converting to [0,1] scale - to be used
        # in acqusition as an initial guess
        x_inc,_,_ = zero_one_scale(self.X_inc[i,:],
                                self.param_bounds[:,0],
                                self.param_bounds[:,1])
        
        # storing incumbent in the original scale
        if self.logscale is not None:
            self.X_inc[i,self.logscale] = np.exp(self.X_inc[i,self.logscale])

        return x_inc        

    def _acquistion(self,x_inc,i):
        # initialize acqusition function
        if self.acq is None:
            self.acq = LCB(self.gp,kappa=self.kappa)
        else:
            self.acq.update(self.gp)
        
        # optimize LCB
        acq_start = time.time()
        x_cand,acq_cand = scipy_minimize(self.acq,
                                        x_inc,
                                        np.zeros((self.gp.n_dim,)),
                                        np.ones((self.gp.n_dim,)),
                                        rng = self.rng,
                                        n_restarts=8+self.gp.n_dim)
        self.acq_time[i] = time.time()-acq_start
        x_cand = self.gp.lower + (self.gp.upper-self.gp.lower)*x_cand

        return x_cand,acq_cand
    
    def save_to_pickle(self,filepath):
        with open(filepath,'wb') as f:
            pickle.dump(self,f)
    
    def _eval_model(self,estimator,train,test,X_alg,y_alg):
        # TODO: add support for unsupervised learning
        # algorithm and corresponding metrics

        if len(X_alg.shape) == 1:
            estimator.fit(X_alg[train],y_alg[train])
            X_alg_test = X_alg[test]
        else:
            estimator.fit(X_alg[train,:],y_alg[train])
            X_alg_test = X_alg[test,:]

        if self.return_prob:
            y_pred = estimator.predict_proba(X_alg_test)
        else:
            y_pred = estimator.predict(X_alg_test)
        return self.metric(y_alg[test],y_pred)
    
    def _fold_eval(self,params,fold_ind,X_alg,y_alg,return_average=False):
        estimator = clone(self.estimator)
        
        params_ = np.copy(params)
        if self.logscale is not None:
            params_[self.logscale] = np.exp(params_[self.logscale])
            
        # set parameters
        for j in np.arange(len(self.param_names)):
            tmp = params_[j]
            if j in self.integer:
                tmp = np.int(np.round(tmp))
                
            estimator.set_params(**{self.param_names[j]:tmp})
            
        time_eval = []
        y_eval = []
        
        for ind in fold_ind:
            start_time = time.time()
            train,test = self.folds[ind]
            y_eval.append(self._eval_model(estimator,train,test,X_alg,y_alg))
            time_eval.append(time.time()-start_time)
        
        if return_average:
            return np.mean(y_eval),np.sum(time_eval)
        else:
            return y_eval,time_eval
    
    def print_and_return(self,x_cand):
        est_cand = self.gp.predict(x_cand,return_std=False)
        if self.logscale is not None:
            x_cand[self.logscale] = np.exp(x_cand[self.logscale])

        if self.verbose >=1 :
            print('')
            print('Number of candidates evaluated.....: %g' % self.X.shape[0])
            print('Number of folds evaluated..........: %g' % self.gp.y.shape[0])
            print('Estimated obj at incumbent.........: %g' % self.y_inc[-1])
            print('Estimated obj at candidate.........: %g' % est_cand)
            print('')
            print('Incumbent at termination:')
            print(self.X_inc[-1,:])
            print('')
            print('Candidate at termination:')
            print(x_cand)
        
        results = dict()
        results["x_best"] = self.X_inc[-1,:]
        results["f_best"] = self.y_inc[-1]
        results["acq_cand"] = self.acq_vec[-1]
        results["x_cand"] = x_cand
        
        return results