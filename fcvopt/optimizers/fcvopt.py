# -*- coding: utf-8 -*-

import numpy as np
import time
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from fcvopt.models.agp import AGPMCMC
from fcvopt.acquisition.lcb import LCB
#from fcvopt.acquisition.improvement import ImprovLCBMCMC
from fcvopt.util.samplers import lh_sampler
from fcvopt.util.wrappers import scipy_minimize 
from fcvopt.util.preprocess import zero_one_scale

class FCVOpt:
    def __init__(self,estimator,param_bounds,metric,cv=10,logscale=None,
                 integer=[],return_prob=None,kernel="matern",
                 n_init=4,min_iter=5,
                 max_iter=10,verbose=0,seed=None):
        self.estimator = estimator
        self.param_names = list(param_bounds.keys())
        self.param_bounds = np.array(list(param_bounds.values()))
        if logscale is not None:
            self.param_bounds[logscale,:] = np.log(self.param_bounds[logscale,:])
        self.logscale = logscale
        self.integer = integer
        self.metric = metric
        
        if return_prob is None:
            self.return_prob = True if self.metric is log_loss else False
        else:
            self.return_prob = return_prob
        
        self.rng = np.random.RandomState(seed=seed)
        if type(cv) is int:
            self.cv = KFold(n_splits=cv,shuffle=True,random_state=self.rng)
        else:
            self.cv = cv
            
        if hasattr(self.estimator,"random_state"):
            self.estimator.random_state = self.rng
        
        self.kernel = kernel
        self.n_init = n_init
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.gp = None
        self.X = None
        self.y = []
        self.eval_time = []
        self.total_time = None
        self.folds = None
    
    def _eval_model(self,estimator,train,test,X_alg,y_alg):
        estimator.fit(X_alg[train,:],y_alg[train])
        if self.return_prob:
            y_pred = estimator.predict_proba(X_alg[test,:])
        else:
            y_pred = estimator.predict(X_alg[test,:])
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
        
            
    def fit(self,X_alg,y_alg):
        
        start_time = time.time()
        if self.gp is None:
            n_dim = self.param_bounds.shape[0]
            self.X = lh_sampler(self.n_init,self.param_bounds[:,0],
                                self.param_bounds[:,1],self.rng)
            self.folds = [ind for ind in self.cv.split(X_alg)]
            self.f_list = np.tile(self.rng.choice(np.arange(self.cv.n_splits),
                                                  size=3,replace=False),
                                  reps=(self.n_init,1)).tolist()
            for i in np.arange(self.n_init):
                tmp1,tmp2 = self._fold_eval(self.X[i,:],
                                            self.f_list[i],
                                            X_alg,y_alg,False)
                self.y.append(tmp1)
                self.eval_time.append(tmp2)
                
            self.gp = AGPMCMC(self.kernel,self.param_bounds[:,0],
                              self.param_bounds[:,1],rng=self.rng)
            self.acq = None
            self.term = None
            
            self.X_inc = np.zeros((self.max_iter,n_dim))
            self.y_inc = np.zeros((self.max_iter,))
            self.acq_vec = np.zeros((self.max_iter,))
            #self.term_vec = np.zeros((self.max_iter,))
            self.sigma_f_vec = np.zeros((self.max_iter,))
            
            # gp timers
            self.mcmc_time = np.zeros((self.max_iter,))
            self.acq_time = np.zeros((self.max_iter,))
            #self.term_time = np.zeros((self.max_iter,))
            
        output_header = '%6s %9s %10s %10s' % \
                    ('iter', 'f_best', 'acq_best',"sigma_f")
        
        for i in range(self.max_iter):
            mcmc_start = time.time()
            self.gp.fit(self.X,self.y,self.f_list)
            self.mcmc_time[i] = time.time()-mcmc_start
            
            self.sigma_f_vec[i] = self.gp.y_scale* \
                            np.sqrt(np.mean([np.exp(model.k1_.theta[-1]) \
                                             for model in self.gp.models]) + \
                                    np.var([model.mu_hat \
                                            for model in self.gp.models]))
            self.X_inc[i,:],self.y_inc[i] = self.gp.get_incumbent()
            
            x_inc,_,_ = zero_one_scale(self.X_inc[i,:],
                                       self.param_bounds[:,0],
                                       self.param_bounds[:,1])
            
            if self.logscale is not None:
                self.X_inc[i,self.logscale] = np.exp(self.X_inc[i,self.logscale])
                
            
#            if self.term is None:
#                self.term = ImprovLCBMCMC(self.gp,x_inc)
#            else:
#                self.term.update(self.gp,x_inc)
#            
#            term_start = time.time()
#            _,term = scipy_minimize(self.term,
#                                    x_inc,
#                                    np.zeros((n_dim,)),
#                                    np.ones((n_dim,)),
#                                    rng = self.rng,
#                                    n_restarts=10)
#            self.term_time[i] = time.time()-term_start
#            self.term_vec[i] = -term
            
            # acquisition function optimization - find candidate
            if self.acq is None:
                self.acq = LCB(self.gp)
            else:
                self.acq.update(self.gp)
            
            acq_start = time.time()
            x_cand,acq_cand = scipy_minimize(self.acq,
                                             x_inc,
                                             np.zeros((n_dim,)),
                                             np.ones((n_dim,)),
                                             rng = self.rng,
                                             n_restarts=10)
            self.acq_time[i] = time.time()-acq_start
            
            x_cand = self.gp.lower + (self.gp.upper-self.gp.lower)*x_cand
            
            dist_cand = np.sum(((self.X-x_cand)/(self.gp.upper-self.gp.lower))**2,
                               axis=1)
            
            new_point = 1 
            point_index = np.argwhere(dist_cand<=1e-3)
            if len(point_index) !=0 :
                point_index = point_index[0,0]
                new_point = 0
                x_cand = self.X[point_index,:].copy()
            
            if self.verbose >= 2:
                if i%10==0:
                    # print header every 10 iterations
                    print(output_header)
                print('%6i %3.3e %3.3e %3.3e' %\
                      (i, self.y_inc[i],acq_cand,self.sigma_f_vec[i]))
                
            self.acq_vec[i] = acq_cand
                        
            if i < self.max_iter:
                # pick fold to evaluate
                f_cand = self._fold_pick(x_cand,new_point,point_index,
                                         X_alg.shape[0])
                
                # evaluate candidate
                y_cand,time_cand = self._fold_eval(x_cand,f_cand,
                                                   X_alg,y_alg,False)
                
                # append observations
                if new_point:
                    self.X = np.vstack((self.X,np.copy(x_cand)))
                    self.y.append(y_cand)
                    self.eval_time.append(time_cand)
                    self.f_list.append(f_cand)
                    
                else:
                    self.y[point_index].extend(y_cand)
                    self.eval_time[point_index].extend(time_cand)
                    self.f_list[point_index].extend(f_cand)
        
        self.total_time = time.time()-start_time
        est_cand = self.gp.predict(x_cand,return_std=False)
        if self.logscale is not None:
            x_cand[self.logscale] = np.exp(x_cand[self.logscale])
            
        # Final output message
        if self.verbose >=1 :
            print('')
            print('Number of candidates evaluated.....: %g' % self.X.shape[0])
            print('Number of folds evaluated..........: %g' % self.gp.y_train.shape[0])
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
    
    def _fold_pick(self,x_cand,new_point,point_index,n_alg):
        f_set = np.arange(len(self.folds))
        if not new_point:
            f_set = np.setdiff1d(f_set,self.f_list[point_index])
            if len(f_set)==1:
                return [f_set[0]]
            elif len(f_set)==0:
                f_cand = len(self.folds) + self.rng.randint(0,self.cv.n_splits)
                self.folds.extend([ind for ind in self.cv.split(np.arange(n_alg))])
                return [f_cand]
                
        f_cand= f_set[np.argmin(self.gp._fold_var(x_cand,f_set))]
        return [f_cand]
    
    def term_crit(self):
        return (self.y_inc-self.acq_vec)/self.sigma_f_vec/self.acq.kappa