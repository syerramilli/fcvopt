import numpy as np
import os 
import time
import torch
import gpytorch
import warnings
import joblib

from .. import kernels
from ..models import GPR
from ..models.emcee_utils import EnsembleMCMC
from ..models.fit_model import fit_model_unconstrained
from ..acquisition.active import ActiveLearningCohn
from .acqfunoptimizer import AcqFunOptimizer
from ..configspace import ConfigurationSpace

from typing import Callable,List,Union,Tuple,Optional,Dict
from collections import OrderedDict
from copy import deepcopy

class ActiveLearning:
    def __init__(
        self,
        obj:Callable,
        config:ConfigurationSpace,
        X_ref:torch.Tensor,
        deterministic:bool=True,
        estimation_method:str='MAP',
        correlation_kernel_class:Optional[str]=None,
        kappa:float=2.,
        verbose:int=0.,
        save_iter:Optional[int]=None,
        save_dir:Optional[int]=None
    ):
        self.obj = obj
        self.config = config
        self.X_ref = X_ref
        self.deterministic = deterministic
        self.estimation_method = estimation_method
        if correlation_kernel_class is None:
            self.correlation_kernel_class = kernels.Matern52Kernel
        else:
            self.correlation_kernel_class = getattr(kernels,correlation_kernel_class)
        
        self.kappa=kappa
        self.verbose=verbose
        # TODO: work on this
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
            key:getattr(self,key) for key in ['train_x','train_y','train_folds']
        },os.path.join(folder,'model_train.pt'))
        # model state dict
        _ = torch.save(self.model.state_dict(),os.path.join(folder,'model_state.pth'))


    def _initialize(self,n_init:Optional[int]=None):
        if self.train_confs is None:
            if n_init is None:
                n_init = len(self.config.quant_index) + 1
            
            self.config.seed(np.random.randint(2e+4))
            self.train_confs = self.config.latinhypercube_sample(n_init)

            for conf in self.train_confs:
                x,y,eval_time = self._evaluate(conf)
                self.train_x.append(x)
                self.train_y.append(y)
                self.obj_eval_time.append(eval_time)
            
            self.train_x = torch.tensor(self.train_x).double()
            self.train_y = torch.tensor(self.train_y).double()
        else:
            # algorithm has been run previously
            # evaluate the next candidate 
            next_conf = self.confs_cand[-1]
            next_x,next_y,eval_time = self._evaluate(next_conf)
            self.train_confs.append(next_conf)
            self.train_y = torch.cat([self.train_y,torch.tensor([next_y]).to(self.train_y)])
            self.train_x = torch.cat([self.train_x,torch.tensor(next_x).to(self.train_x).reshape(1,-1)])
            self.obj_eval_time.append(eval_time)
    
    def _evaluate(self,conf,**kwargs):
        start_time = time.time()
        y = self.obj(conf.get_dictionary(),**kwargs)
        eval_time = time.time()-start_time
        return conf.get_array(),y,eval_time
    
    def _fit_model(self) -> None:
        # construct model
        self.model = self._construct_model()

        start_time = time.time()
        if self.estimation_method == 'MCMC':
            num_steps = 200 if self.initial_params is not None else 1000
            burnin = 200 if self.initial_params is not None else 1000
            mcmc = EnsembleMCMC(self.model,burnin,num_steps,p0=self.initial_params)
            self.initial_params = mcmc.run(progress=True)
            
        elif self.estimation_method == 'MAP':
            if self.initial_params is not None:
                self.model.initialize(**self.initial_params)

            _ = fit_model_unconstrained(
                model = self.model,
                num_restarts = 9
            )

            self.initial_params = OrderedDict()
            # disable model gradients
            for name,parameter in self.model.named_parameters():
                parameter.requires_grad_(False)
                self.initial_params[name] = parameter
        # generate model cache
        self.model.eval()
        with torch.no_grad():
            _ = self.model(self.train_x[[0],:])

        self.fit_time.append(time.time()-start_time)
    
    def _construct_model(self):
        noise = 1e-4 if self.deterministic else 1e-2
        return GPR(
            train_x = self.train_x,
            train_y = self.train_y,
            correlation_kernel_class=self.correlation_kernel_class,
            noise=noise,
            fix_noise=self.deterministic,
            estimation_method=self.estimation_method
        ).double()
    
    def _calculate_prior_sigma(self) -> float:
        var_vec = self.model.covar_module.outputscale*(self.model.y_std**2)
        
        if self.estimation_method  == 'MAP':
            return var_vec.sqrt().item()
        
        # else MCMC
        mean_vec = self.model.mean_module.constant*self.model.y_std + self.model.y_mean
        return torch.sqrt(mean_vec.var()+var_vec.mean()).item()
    
    def _acquisition(self) -> None:
        acqobj = ActiveLearningCohn(self.model,self.X_ref)
        
        acqopt = AcqFunOptimizer(
            acq_fun=acqobj,
            ndim = len(self.config.quant_index),
            num_starts = 10*len(self.config.quant_index),
            x0=None,
            num_jobs=1 # TODO: add support for parallelization
        )
        start_time = time.time()
        next_x = acqopt.run()
        self.acqopt_time.append(time.time()-start_time)
        self.confs_cand.append(self.config.get_conf_from_array(next_x))
        self.acq_vec.append(acqopt.obj_sign*acqopt.f_inc)
