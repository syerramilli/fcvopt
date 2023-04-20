import numpy as np
import os 
import time
import torch
import gpytorch
import warnings
import joblib

from ..models import GPR
from ..fit.mll_scipy import fit_model_scipy
from ..configspace import ConfigurationSpace
from ..acquisition.active import ActiveLearningCohn

from .acqfunoptimizer import AcqFunOptimizer

from typing import Callable,List,Union,Tuple,Optional,Dict
from collections import OrderedDict
from copy import deepcopy

class ActiveLearning:
    def __init__(
        self,
        obj:Callable,
        config:ConfigurationSpace,
        X_ref:torch.Tensor,
        n_jobs:int=1,
        verbose:int=1,
        save_iter:Optional[int]=None,
        save_dir:Optional[int]=None,
    ):
        self.obj = obj
        self.config = config
        self.X_ref = X_ref
        self.n_jobs = n_jobs
        self.verbose=verbose
        self.save_iter = save_iter
        self.save_dir = save_dir

        # initialize objects
        self.model = None
        self.train_confs = None
        self.train_x = []
        self.train_y = []
        self.confs_cand = []
        self.acq_vec = []
        self.fit_time = []
        self.acqopt_time = []
        self.obj_eval_time = []
        # mcmc parameters
        self.initial_params = None
    
    def run(self,n_iter:int,n_init:Optional[int]=None) -> Dict:
        output_header = '%6s %10s' % ('iter', 'max_var')
        for i in range(n_iter):
            # either initialize observations or evaluate the next candidate
            self._initialize(n_init)

            # fit model
            self._fit_model()
        
            # acquisition find next candidate
            self._acquisition()

            if self.save_iter and self.save_dir:
                if i % self.save_iter == 0:
                    self.save_to_file(self.save_dir)

            # update verbose statements
            if self.verbose >= 2:
                if i%10 == 0:
                    # print header every 19 iterations
                    print(output_header)
                print('%6i %10.3e' % (i,self.acq_vec[-1]))
        
        if self.verbose >= 1:

            print('')
            print('Number of candidates evaluated.....: %g' % len(self.train_confs))
            print('Posterior variance at candidate....: %g' % self.acq_vec[-1])
            print('')
            print('Candidate at termination:')
            print(self.confs_cand[-1])

    def save_to_file(self,folder):
        #  optimization statistics
        stat_keys = [
            'acq_vec','confs_cand',
            'fit_time','acqopt_time','obj_eval_time',
        ]
        stats = {
            key:getattr(self,key) for key in stat_keys
        }
        joblib.dump(stats,os.path.join(folder,'stats.pkl'))
        # Observations
        _ = torch.save({
            key:getattr(self,key) for key in ['train_x','train_y']
        },os.path.join(folder,'model_train.pt'))
        # model state dict
        _ = torch.save(self.model.state_dict(),os.path.join(folder,'model_state.pth'))


    def _initialize(self,n_init:Optional[int]=None):
        if self.train_confs is None:
            if n_init is None:
                n_init = len(self.config.quant_index) + 1
            
            self.config.seed(np.random.randint(2e+4))
            self.train_confs = self.config.latinhypercube_sample(n_init)

            evaluations = self._evaluate_confs(self.train_confs)
            for x,y,eval_time in evaluations:
                self.train_x.append(x)
                self.train_y.append(y)
                self.obj_eval_time.append(eval_time)
            
            self.train_x = torch.tensor(self.train_x).double()
            self.train_y = torch.tensor(self.train_y).double()
        else:
            # algorithm has been run previously
            # evaluate the next candidate 
            next_confs_list = self.confs_cand[-1]
            evaluations = self._evaluate_confs(next_confs_list)
            
            for next_conf,(next_x,next_y,eval_time) in zip(next_confs_list,evaluations):
                self.train_confs.append(next_conf)
                self.train_y = torch.cat([self.train_y,torch.tensor([next_y]).to(self.train_y)])
                self.train_x = torch.cat([self.train_x,torch.tensor(next_x).to(self.train_x).reshape(1,-1)])
                self.obj_eval_time.append(eval_time)
    
    def _evaluate(self,conf,**kwargs):
        start_time = time.time()
        y = self.obj(conf.get_dictionary(),**kwargs)
        eval_time = time.time()-start_time
        return conf.get_array(),y,eval_time

    def _evaluate_confs(self,confs_list,**kwargs):
        if self.n_jobs > 1 and len(confs_list) > 1:
            # enable parallel evaulations
            evaluations = joblib.Parallel(n_jobs=self.n_jobs,verbose=0)(
                joblib.delayed(self._evaluate)(conf,**kwargs) for conf in confs_list
            )
        else:
            # can add logging here
            evaluations = [None]*len(confs_list)
            for i,conf in enumerate(confs_list):
                evaluations[i] = self._evaluate(conf,**kwargs)
        
        return evaluations
    
    def _fit_model(self) -> None:
        # construct model
        self.model = self._construct_model()

        start_time = time.time()
        start_time = time.time()
        if self.initial_params is not None:
            self.model.initialize(**self.initial_params)

        _ = fit_model_scipy(model = self.model,num_restarts = 5)
            
        self.fit_time.append(time.time()-start_time)

        self.initial_params = OrderedDict()
        for name,parameter in self.model.named_parameters():
                self.initial_params[name] = parameter.detach().clone()
            
        # generate model cache
        self.model.eval()
        with torch.no_grad():
            _ = self.model(self.train_x[[0],:])
    
    def _construct_model(self):
        return GPR(
            train_x = self.train_x,
            train_y = self.train_y,
        ).double()
    
    def _acquisition(self) -> None:
        acqobj = ActiveLearningCohn(self.model, self.X_ref)
        
        acqopt = AcqFunOptimizer(
            acq_fun=acqobj,
            ndim = len(self.config.quant_index),
            num_starts = 10*len(self.config.quant_index),
            x0=None,
            num_jobs=1,
        )
        start_time = time.time()
        next_x = acqopt.run()
        self.acqopt_time.append(time.time()-start_time)
        self.confs_cand.append(self.config.get_conf_from_array(next_x))
        self.acq_vec.append(acqopt.obj_sign*acqopt.f_inc)