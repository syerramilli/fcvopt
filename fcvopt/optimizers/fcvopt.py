# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import pickle
from sklearn.base import clone
from sklearn.model_selection import KFold

from fcvopt.optimizers.BayesOpt import BayesOpt
from fcvopt.models.agp import AGP
from fcvopt.acquisition import LCB
from fcvopt.util.samplers import lh_sampler
from fcvopt.util.wrappers import scipy_minimize 
from fcvopt.util.preprocess import zero_one_scale

class FCVOpt(BayesOpt):
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
        Confidence level for the lower confidence bound. Default
        value of 2 works for most cases
    
    n_init: int (defult: 4)
        The number of points for the intial design. The higher the number, 
        the better the model, but the higher the computational expense. 
        Recommendation: use D+1 initial configurations, where D is the 
        number of hyperparameters to be optimized
    
    max_iter: int (default: 30)
        The number of iterations, excluding initial design. 
    
    verbose: int (default: 0)
        Parameter controlling the amount of output printed
        - >= 2: print progress at ech iteration
        - == 1: print statistics at termination
        - == 0: no output
    
    seed: int or None (default: None)
        The seed to be used for the random number generator

    save_iter: int or None (default: None)
        The number of iterations at which the model is saved to disk 
        periodically. Useful when max_iter is large. See documentation
        for the `resume` method to resume iterations for interrupted/
        incomplete runs
    
    save_dir: None
        The directory to periodically save progress. Useful only
        when `save_iter` is not None.
    '''
    def __init__(self,estimator,param_bounds,metric,n_folds=5,logscale=None,
                 integer=[],return_prob=False,kernel="matern",kappa=2,
                 n_init=4,max_iter=30,verbose=0,seed=None,save_iter=None,
                 save_dir=None):
        
        super(FCVOpt,self).__init__(estimator,param_bounds,metric,n_folds,logscale,
                                    integer,return_prob,kernel,kappa,n_init,
                                    max_iter,verbose,seed,save_iter,save_dir)
    
    def run(self,X_alg,y_alg):
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
            self.X_inc[i,:],self.y_inc[i] = self.gp.get_incumbent()
            
            # converting to [0,1] scale - to be used
            # in acqusition as an initial guess
            x_inc,_,_ = zero_one_scale(self.X_inc[i,:],
                                       self.param_bounds[:,0],
                                       self.param_bounds[:,1])
            
            # storing incumbent in the original scale
            if self.logscale is not None:
                self.X_inc[i,self.logscale] = np.exp(self.X_inc[i,self.logscale])
            
            ########## Acquisition ##########
            # acquisition function optimization - find candidate
            x_cand, acq_cand = self._acquistion(x_inc,i)
            
            # checking if point is close to existing design points
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
                    filepath = os.path.join(self.save_dir,"iter_"+str(i)+".pkl")
                    self.save_to_pickle(filepath)
                
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
        
        self.total_time = time.time()-start_time
            
        # Final output message
        return self.print_and_return(x_cand)
    
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
    
    def resume(self,X_alg,y_alg):
        '''
        Resume interrupted runs. The algorithm resumes from the last
        completed iteration.
        
        Parameters
        -------------
        X_alg: 2D array
            The features/covariates for the supervised learning algorithm
            
        y_alg: 1D array
            The output for the supervised learning algorith,
        '''
        # determine last iteration
        last_iter = next((i-1 for i, x in enumerate(self.y_inc) if np.abs(x) < 1e-8), None)
        
        N_gp = self.gp.Kinv_[0].shape[0] # number of observations in fitted GP model
        n_obs_y = np.sum([len(tmp) for tmp in self.y]) # number of points added
        
        if N_gp < n_obs_y:
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
            self.X_inc[i,:],self.y_inc[i] = self.gp.get_incumbent()
            
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
                    filepath = os.path.join(self.save_dir,"iter_"+str(i)+".pkl")
                    self.save_to_pickle(filepath)
            
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
        
        # Final output message
        return self.print_and_return(x_cand)