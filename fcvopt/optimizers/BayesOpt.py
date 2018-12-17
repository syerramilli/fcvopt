# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,log_loss
from sklearn.utils.validation import check_X_y

class BayesOpt:
    def __init__(self,estimator,param_bounds,metric,cv=10,logscale=None,
                 kernel="matern",n_init=2,min_iter=5,max_iter=10,verbose=0,
                 seed=None):
        self.estimator = estimator
        self.param_names = list(param_bounds.keys())
        self.param_bounds = np.array(list(param_bounds.values()))
        if logscale is not None:
            self.param_bounds[:,logscale] = np.log(self.param_bounds[:,logscale])
        self.logscale = logscale
        self.metric = metric
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
        
        self.cv_model = None
        self.X = None
        self.y = None
        self.folds = None
        
    def _generate_grid(self,n_samples):
        out = (self.random_state.uniform(self.param_bounds[i,0],
                                         self.param_bounds[i,1],n_samples) 
                for i in range(self.param_bounds.shape[1]))
        return np.column_stack(out)
    
    
    def _eval_model(self,estimator,train,test,X_alg,y_alg):
        estimator.fit(X_alg[train,:],y_alg[train])
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
        
        if self.cv_model is None:
            d = self.param_bounds.shape[1]
            self.X = self._generate_grid(self.n_init*d)
            self.folds = np.array([ind for ind in self.cv.split(X_alg)])
            self.y = [self._fold_eval(self.X[i,:],
                                      np.arange(self.folds.shape[0]),
                                      X_alg,y_alg,True)\
            for i in np.arange(self.n_init*d)]
            self.cv_model = GPMCMC(self.kernel,self.param_bounds[:,0],
                                   self.param_bounds[:,1])
            self.X_inc = np.zeros((self.max_iter,d))
            self.y_inc = np.zeros((self.max_iter,))
        
        for i in range(self.max_iter):
            self.cv_model.fit(self.X,self.y)
            self.X_inc[i,:],self.y_inc[i] = self.cv_model.get_incumbent()
            
            
        
        
        