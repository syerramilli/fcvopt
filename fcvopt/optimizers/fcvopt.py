# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import pickle
from sklearn.base import clone
from sklearn.model_selection import KFold

from fcvopt.models.agp import AGP
from fcvopt.acquisition import LCB
from fcvopt.util.samplers import lh_sampler
from fcvopt.util.wrappers import scipy_minimize 
from fcvopt.util.preprocess import zero_one_scale

class FCVOpt:
    '''
    Fast cross-validation for optimizing hyperparameters of supervised
    learning algorithms
    
    Parameters
    --------------
    estimator: estimator object
        An object implementing the scikit-learn estimator interface
    
    param_bounds: dict
        Dictionary with hyperparameter names as keys and lists of lower 
        and upper bounds for each hyperparameter. Currently only
        numerical-valued hyperparameters are supported
        
    metric: callable
        Callable to evaluate the predictions on the test set. Must accept
        two arguments- `y_true` and `y_pred`.
    
    n_folds: int (default: 5)
        The number of folds in the fold partition. Either 5 or 10
        is recommended
        
    logscale: 1D array or None (default: None)
        Array of indices, corresponding to the order in `param_bounds`,
        of those hyperparameters which need to be optimized in logscale
        
    integer: 1D array or None (default:None)
        Array of indices, corresponding to the order in `param_bounds`,
        of those hyperparameters which are integer-valued. Integrality 
        will be enforced when evaluating the configurations
        
    return_prob: boolean (default: False)
        Logical indicating whether the metric returns probablity
        estimates out of a classifer
        
    kernel: string
        Name of the kernel. Current options are "gaussian" (Gaussian or 
        RBF Kernel) and "matern" (Matern 5/2 kernel). These kernels are 
        implemented in scikit-learn
        
    kappa: int (default: 2)
    
    n_init: int (defult: 4)
    
    max_iter: int (default: 10)
    
    verbose: int (default: 10)
    
    seed: int or None (default: None)

    save_iter: int or None (default: None)
    
    save_dir: None    
    '''
    def __init__(self,estimator,param_bounds,metric,n_folds=5,logscale=None,
                 integer=[],return_prob=False,kernel="matern",kappa=2,
                 n_init=4,max_iter=10,verbose=0,seed=None,save_iter=None,
                 save_dir=None):
        self.estimator = estimator
        self.param_names = list(param_bounds.keys())
        self.param_bounds = np.array(list(param_bounds.values()))
        if logscale is not None:
            self.param_bounds[logscale,:] = np.log(self.param_bounds[logscale,:])
        self.logscale = logscale
        self.integer = integer
        self.metric = metric
        self.return_prob = return_prob
        
        self.rng = np.random.RandomState(seed=seed)
        self.cv = KFold(n_splits=n_folds,shuffle=True,random_state=self.rng)
        self.n_folds = n_folds
        
            
        if hasattr(self.estimator,"random_state"):
            self.estimator.random_state = self.rng
        
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
    
    def _eval_model(self,estimator,train,test,X_alg,y_alg):
        # TODO: add support for unsupervised learning
        # algorithm and corresponding metrics
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
        '''
        Optimize hyperparameters of the specfied learning algorithm on 
        the given dataset
        
        Parameters
        -------------
        X_alg: 2D array
            The features/covariates for the supervised learning algorithm
            
        y_alg: 1D array
            The output for the supervised learning algorith,
        '''
        start_time = time.time()
        if self.gp is None:
            # TODO: add else case to run optimization
            # for additional iterations 
            n_dim = self.param_bounds.shape[0]
            
            # initial configuration sampled using LHS
            if type(self.n_init) is int:
                self.X = lh_sampler(self.n_init,self.param_bounds[:,0],
                                    self.param_bounds[:,1],self.rng)
            else:
                self.X = self.n_init
                self.n_init = self.X.shape[0]
            
            # Generate fold partition and assign folds at random
            # to the initial configurations
            self.folds = [ind for ind in self.cv.split(X_alg)]
            self.f_list =[self.rng.choice(self.cv.n_splits,1,
                                           replace=False).tolist() for _ in range(self.n_init)]
            
            # evaluate initial hyperparameter configurations
            for i in np.arange(self.n_init):
                tmp1,tmp2 = self._fold_eval(self.X[i,:],
                                            self.f_list[i],
                                            X_alg,y_alg,False)
                self.y.append(tmp1)
                self.eval_time.append(tmp2)
            
            # number of AGP hyperparametes to store 
            # also servers as the number of walkers
            n_hypers = (n_dim+5)*3
            if n_hypers % 2 == 1:
                n_hypers +=1
            
            # initialize AGP model
            self.gp = AGP(self.kernel,self.param_bounds[:,0],
                              self.param_bounds[:,1],
                              n_hypers=n_hypers,
                              chain_length=100,rng=self.rng,
                              burnin_length=100)
            self.acq = None
            self.term = None
            
            self.X_inc = np.zeros((self.max_iter,n_dim))
            self.y_inc = np.zeros((self.max_iter,))
            self.acq_vec = np.zeros((self.max_iter,))
            self.sigma_f_vec = np.zeros((self.max_iter,))
            
            # gp timers
            self.mcmc_time = np.zeros((self.max_iter,))
            self.acq_time = np.zeros((self.max_iter,))
            
        output_header = '%6s %9s %10s %10s' % \
                    ('iter', 'f_best', 'acq_best',"sigma_f")
        
        for i in range(self.max_iter):
            
            ########## "Fit" AGP model ########## 
            mcmc_start = time.time()
            self.gp.fit(self.X,self.y,self.f_list)
            self.mcmc_time[i] = time.time()-mcmc_start
            
            
            self.sigma_f_vec[i] = \
                            np.sqrt(np.mean([np.exp(self.gp.k1_[i].theta[-1]) \
                                             for i in range(self.gp.n_hypers)]) + \
                                    np.var(self.gp.mu_))
            
            ########## Find incumbent ##########
            self.X_inc[i,:],self.y_inc[i],_ = self.gp.get_incumbent()
            
            # converting to [0,1] scale - to be used
            # in acqusition as an initial guess
            x_inc,_,_ = zero_one_scale(self.X_inc[i,:],
                                       self.param_bounds[:,0],
                                       self.param_bounds[:,1])
            
            # storing incumbent in the original scale
            if self.logscale is not None:
                self.X_inc[i,self.logscale] = np.exp(self.X_inc[i,self.logscale])
            
            ########## Acquisition ##########
            if self.acq is None:
                self.acq = LCB(self.gp,kappa=self.kappa)
            else:
                self.acq.update(self.gp)
            
            # optimize LCB
            acq_start = time.time()
            x_cand,acq_cand = scipy_minimize(self.acq,
                                             x_inc,
                                             np.zeros((n_dim,)),
                                             np.ones((n_dim,)),
                                             rng = self.rng,
                                             n_restarts=9)
            self.acq_time[i] = time.time()-acq_start
            
            
            # checking if point is close to existing design points
            x_cand = self.gp.lower + (self.gp.upper-self.gp.lower)*x_cand
            dist_cand = np.sum(np.abs(self.X-x_cand)/(self.gp.upper-self.gp.lower),
                               axis=1)
            new_point = 1
            point_index = np.argmin(dist_cand)
            if np.round(dist_cand[point_index],2) <= 1e-2+1e-8:
                new_point = 0
                x_cand = self.X[point_index,:].copy()
                acq_cand = self.acq(x_cand,scaled=False)
            
            ########## Printing updates ##########
            if self.verbose >= 2:
                if i%10==0:
                    # print header every 10 iterations
                    print(output_header)
                print('%6i %3.3e %3.3e %3.3e' %\
                      (i, self.y_inc[i],acq_cand,self.sigma_f_vec[i]))
            
            self.acq_vec[i] = acq_cand
            
            ########## Saving progress to disk ##########
            if self.save_iter is not None:
                if (i+1)%self.save_iter == 0:
                    fname = os.path.join(self.save_dir,"iter_"+str(i)+".pkl")
                    with open(fname,"wb") as f:
                        pickle.dump(self,f)
            
            ########## Evaluate new configurations ##########
            if i < self.max_iter-1:
                # evaluate new configurations to be evaluted 
                # provided the next iteration is still allowed
                
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
        
        ########## Tidying up ##########
        self.total_time = time.time()-start_time
        est_cand = self.gp.predict(x_cand,return_std=False)
        if self.logscale is not None:
            x_cand[self.logscale] = np.exp(x_cand[self.logscale])
            
        ########## Final output message ##########
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
        
        
        ########## Return acqusision and candidate as output ##########
        results = dict()
        results["x_best"] = self.X_inc[-1,:]
        results["f_best"] = self.y_inc[-1]
        results["acq_cand"] = self.acq_vec[-1]
        results["x_cand"] = x_cand
        
        return results
    
    def _fold_pick(self,x_cand,new_point,point_index,n_alg):
        """
        Pick fold (index) at the candidate point
        
        Parameters
        -------------
        x_cand: 1-D array, shape = (n_dim,)
            The candidate at which the fold needs to be picked
            
        new_point: boolean
            Boolean variable indicating whether the candidate has
            already been sampled or not
            
        point_index: int
            If new_point is True, the index of the corresponding 
            observation in the configuration matrix
        
        n_alg: int
            Number of samples in the dataset for the supervised
            learning algorithm
        """
        f_set = np.arange(self.cv.n_splits)
        if not new_point:
            f_set = np.setdiff1d(f_set,self.f_list[point_index])
            if len(f_set)==1:
                return [f_set[0]]
            elif len(f_set)==0:
                # if all of the existing folds from the original 
                # partition are evaluated at this configuration,
                # sample from another partition. The uncertainty
                # reduction criteria no longer applies
                if len(self.f_list[point_index]) % len(self.folds) == 0:
                    # generate new partition if all partitons
                    # are exhausted at this configuration and
                    # assign fold at random from this new partition
                    f_cand = len(self.folds) + self.rng.randint(0,self.cv.n_splits)
                    self.folds.extend([ind for ind in self.cv.split(np.arange(n_alg))])
                else:
                    # partitions are not exhausted
                    f_set = np.setdiff1d(np.arange(len(self.folds)),
                                         self.f_list[point_index])
                    f_cand = self.rng.choice(f_set,1)[0]
                return [f_cand]
        
        # folds from the original partition are not exhausted 
        f_cand= f_set[np.argmin(self.gp._fold_var(x_cand,f_set))]
        return [f_cand]
    
    def term_crit(self):
        """
        Returns an array of termination metrics at the incumbent
        of each iteration
        """
        return (self.y_inc-self.acq_vec)/self.sigma_f_vec/self.acq.kappa
    
    def resume(self,X_alg,y_alg):
        # determine last iteration
        last_iter = next((i-1 for i, x in enumerate(self.y_inc) if np.abs(x) < 1e-8), None)
        
        n_obs_y = np.sum([len(tmp) for tmp in self.y]) # number of points added
        n_eval = last_iter + self.n_init  # number of points evaluated by GP model
        
        if n_eval == n_obs_y + 1:
            # interrupted while updating model
            # a new iteration has just begun
            print("**** Interrupted when updating model ****\n")
            model_flag = True
            last_iter = last_iter+1
        else:
            # model interrupted after updating model
            # need to repeat from acqusition stage since
            # candidate point isn't stored
            print("**** Interrupted after updating model ****\n")
            model_flag = False
            
        
        ######### Recalculations #############
        n_dim = self.param_bounds.shape[0]
        output_header = '%6s %9s %10s %10s' % \
                    ('iter', 'f_best', 'acq_best',"sigma_f")          
        if self.verbose >= 2:
            print(output_header)
        
        for i in np.arange(last_iter,self.max_iter):
            ########## "Fit" AGP model ##########
            no_update_flag = (i==last_iter) and not model_flag
            if not no_update_flag:
                mcmc_start = time.time()
                self.gp.fit(self.X,self.y,self.f_list)
                self.mcmc_time[i] = time.time()-mcmc_start
            
            self.sigma_f_vec[i] = \
                            np.sqrt(np.mean([np.exp(self.gp.k1_[i].theta[-1]) \
                                             for i in range(self.gp.n_hypers)]) + \
                                    np.var(self.gp.mu_))
            
            ########## Find incumbent ##########
            self.X_inc[i,:],self.y_inc[i],_ = self.gp.get_incumbent()
            
            # converting to [0,1] scale - to be used
            # in acqusition as an initial guess
            x_inc,_,_ = zero_one_scale(self.X_inc[i,:],
                                       self.param_bounds[:,0],
                                       self.param_bounds[:,1])
            
            # storing incumbent in the original scale
            if self.logscale is not None:
                self.X_inc[i,self.logscale] = np.exp(self.X_inc[i,self.logscale])
            
            ########## Acquisition ##########
            self.acq.update(self.gp)
            
            # optimize LCB
            acq_start = time.time()
            x_cand,acq_cand = scipy_minimize(self.acq,
                                             x_inc,
                                             np.zeros((n_dim,)),
                                             np.ones((n_dim,)),
                                             rng = self.rng,
                                             n_restarts=9)
            self.acq_time[i] = time.time()-acq_start
            
            
            # checking if point is close to existing design points
            x_cand = self.gp.lower + (self.gp.upper-self.gp.lower)*x_cand
            dist_cand = np.sum(np.abs(self.X-x_cand)/(self.gp.upper-self.gp.lower),
                               axis=1)
            new_point = 1
            point_index = np.argmin(dist_cand)
            if np.round(dist_cand[point_index],2) <= 1e-2+1e-8:
                new_point = 0
                x_cand = self.X[point_index,:].copy()
                acq_cand = self.acq(x_cand,scaled=False)
            
            ########## Printing updates ##########
            if self.verbose >= 2:
                if i%10==0:
                    # print header every 10 iterations
                    print(output_header)
                print('%6i %3.3e %3.3e %3.3e' %\
                      (i, self.y_inc[i],acq_cand,self.sigma_f_vec[i]))
            
            self.acq_vec[i] = acq_cand
            
            ########## Saving progress to disk ##########
            if self.save_iter is not None:
                if (i+1)%self.save_iter == 0:
                    fname = os.path.join(self.save_dir,"iter_"+str(i)+".pkl")
                    with open(fname,"wb") as f:
                        pickle.dump(self,f)
            
            ########## Evaluate new configurations ##########
            if i < self.max_iter-1:
                # evaluate new configurations to be evaluted 
                # provided the next iteration is still allowed
                
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
        
        ########## Tidying up ##########
        #self.total_time = time.time()-start_time
        est_cand = self.gp.predict(x_cand,return_std=False)
        if self.logscale is not None:
            x_cand[self.logscale] = np.exp(x_cand[self.logscale])
            
        ########## Final output message ##########
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
        
        
        ########## Return acqusision and candidate as output ##########
        results = dict()
        results["x_best"] = self.X_inc[-1,:]
        results["f_best"] = self.y_inc[-1]
        results["acq_cand"] = self.acq_vec[-1]
        results["x_cand"] = x_cand
        
        return results