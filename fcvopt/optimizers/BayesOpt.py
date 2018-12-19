# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss


# from fcvopt.models.gp import GPMCMC
# from cvopt.acquisition.lcb import LCB
# from fcvopt.util.samples import lh_sampler
# from fcvopt.util.wrappers import scipy_minimize 


class BayesOpt:
    def __init__(self,estimator,param_bounds,metric,cv=10,logscale=None,
                 return_prob=None,kernel="matern",n_init=3,min_iter=5,
                 max_iter=10,verbose=0,seed=None):
        self.estimator = estimator
        self.param_names = list(param_bounds.keys())
        self.param_bounds = np.array(list(param_bounds.values()))
        if logscale is not None:
            self.param_bounds[:,logscale] = np.log(self.param_bounds[:,logscale])
        self.logscale = logscale
        self.metric = metric
        
        if return_prob is None:
            self.return_prob = True if self.metric is log_loss else False
        else:
            self.return_prob = return_prob
        
        self.random_state = np.random.RandomState(seed=seed)
        if type(cv) is int:
            self.cv = KFold(n_splits=cv,random_state=self.random_state)
        else:
            self.cv = cv
        
        self.kernel = kernel
        self.n_init = n_init
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.gp = None
        self.X = None
        self.y = None
        self.folds = None
    
    def _eval_model(self,estimator,train,test,X_alg,y_alg):
        estimator.fit(X_alg[train,:],y_alg[train])
        if self.return_prob:
            y_pred = estimator.predict_proba(X_alg[test,:])
        else:
            y_pred = estimator.predict(X_alg[test,:])
        return self.metric(y_alg[test],y_pred)
    
    def _fold_eval(self,params,fold_ind,X_alg,y_alg,return_average=False):
        estimator = deepcopy(self.estimator)
        
        params_ = np.copy(params)
        if self.logscale is not None:
            params_[self.logscale] = np.exp(params_[self.logscale])
            
        # set parameters
        for j in np.arange(len(self.param_names)):
            setattr(estimator,self.param_names[j],params_[j])
            
        y_eval = [self._eval_model(estimator,train,test,X_alg,y_alg) \
                  for train,test in self.folds[fold_ind]]
        
        if return_average:
            return np.mean(y_eval)
        else:
            return np.array(y_eval)
        
            
    def fit(self,X_alg,y_alg):
        
        if self.gp is None:
            n_dim = self.param_bounds.shape[1]
            self.X = lh_sampler(self.n_init,self.param_bounds[:,0],
                                self.param_bounds[:,1],self.random_state)
            self.folds = np.array([ind for ind in self.cv.split(X_alg)])
            self.y = [self._fold_eval(self.X[i,:],
                                      np.arange(self.folds.shape[0]),
                                      X_alg,y_alg,True)\
            for i in np.arange(self.n_init)]
            self.gp = GPMCMC(self.kernel,self.param_bounds[:,0],
                                   self.param_bounds[:,1])
            self.acq = None
            
            self.X_inc = np.zeros((self.max_iter,n_dim))
            self.y_inc = np.zeros((self.max_iter,))
            self.acq_vec = np.zeros((self.max_iter,))
            
        output_header = '%6s %9s %9s' % \
                    ('iter', 'f_best', 'acq_best')
        for i in range(self.max_iter):
            self.gp.fit(self.X,self.y)
            self.X_inc[i,:],self.y_inc[i] = self.gp.get_incumbent()
            
            if self.logscale is not None:
                self.X_inc[i,self.logscale] = np.exp(self.X_inc[i,self.logscale])
            
            # acquisition function optimization - find candidate
            if self.acq is None:
                self.acq = LCBMCMC(self.gp)
            else:
                self.acq.update(self.gp)
                
            x_cand,acq_cand = scipy_minimize(self.acq,
                                             np.zeros((n_dim,)),
                                             np.ones((n_dim,)),
                                             rng = self.random_state)   
            x_cand = self.gp.lower + (self.gp.upper-self.gp.lower)*x_cand
            
            if self.verbose >= 2:
                if i%10==0:
                    # print header every 10 iterations
                    print(output_header)
                print('%6i %3.3e %3.3e' %\
                      (i, self.y_inc[i],acq_cand))
                
            self.acq_vec[i] = acq_cand
                        
            if i < self.max_iter-1:
                # evaluate candidate
                y_cand = self._fold_eval(x_cand,np.arange(self.folds.shape[0]),
                                     X_alg,y_alg,True)
                # append observations
                self.X = np.vstack((self.X,x_cand))
                self.y = np.append(self.y,y_cand)
                
        
        est_cand = self.gp.predict(x_cand,return_std=False)
        if self.logscale is not None:
            x_cand[self.logscale] = np.exp(x_cand[self.logscale])
            
        # Final output message
        if self.verbose >=1 :
            print('')
            print('Number of candidates evaluated.....: %g' % self.X.shape[0])
            print('Number of folds in k-fold CV.......: %g' % self.cv.n_splits)
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
        
        
        