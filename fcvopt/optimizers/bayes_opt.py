import numpy as np
import os 
import time
import torch
import gpytorch
import warnings

from .. import kernels
from ..models import GPR
from ..models.mcmc_utils import mcmc_run
from ..models.fit_model import fit_model_unconstrained
from ..acquisition import LowerConfidenceBound,LowerConfidenceBoundMCMC
from .acqfunoptimizer import AcqFunOptimizer
from ..configspace import ConfigurationSpace

from typing import Callable,List,Union,Tuple,Optional,Dict
from collections import OrderedDict
from copy import deepcopy

class BayesOpt:
    def __init__(
        self,
        obj:Callable,
        config:ConfigurationSpace,
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
        self.confs_inc = []
        self.confs_cand = []
        self.f_inc_obs = []
        self.f_inc_est = []
        self.acq_vec = []
        self.sigma_vec = []
        self.fit_time = []
        self.acqopt_time = []
        self.obj_eval_time = []
        # mcmc parameters
        self.initial_params = None
    
    def run(self,n_iter:int,n_init:Optional[int]=None) -> Dict:

        output_header = '%6s %10s %10s %10s %12s' % \
                    ('iter', 'f_inc_obs', 'f_inc_est','acq_cand',"term_metric")
        for i in range(n_iter):
            # either initialize observations or evaluate the next candidate
            self._initialize(n_init)

            # fit model and find incumbent
            self._fit_model_and_find_inc()
        
            # acquisition find next candidate
            self._acquisition()

            # update verbose statements
            if self.verbose >= 2:
                if i%10 == 0:
                    # print header every 19 iterations
                    print(output_header)
                term_metric = (self.f_inc_est[-1]-self.acq_vec[-1])/self.sigma_vec[-1]/self.kappa
                print('%6i %10.3e %10.3e %10.3e %12.4f' %\
                      (i, self.f_inc_obs[-1],self.f_inc_est[-1],
                      self.acq_vec[-1],term_metric))
        
        if self.verbose >= 1:
            est_cand = self.model.predict(
                torch.tensor(self.confs_cand[-1].get_array())
                .to(self.train_x)
                .view(1,-1)
            )
            print('')
            print('Number of candidates evaluated.....: %g' % len(self.train_confs))
            print('Observed obj at incumbent..........: %g' % self.f_inc_obs[-1])
            print('Estimated obj at incumbent.........: %g' % self.f_inc_est[-1])
            print('Estimated obj at candidate.........: %g' % est_cand)
            print('')
            print('Incumbent at termination:')
            print(self.confs_inc[-1])
            print('')
            print('Candidate at termination:')
            print(self.confs_cand[-1])
        
        # return best configuration and other statistics
        results = OrderedDict()
        results['conf_inc'] = self.confs_inc[-1]
        results['f_inc_obs'] = self.f_inc_obs[-1]
        results['f_inc_est'] = self.f_inc_est[-1]

        return results

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
    
    def _fit_model_and_find_inc(self) -> None:
        # construct model
        self.model = self._construct_model()

        start_time = time.time()
        if self.estimation_method == 'MCMC':
            _ = mcmc_run(
                model=self.model,
                step_size=1.0,
                adapt_step_size=True,
                initial_params=self.initial_params,
                disable_progbar=False,
                num_samples=50,
                warmup_steps=100,
                num_model_samples=50
            )
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

        self.fit_time.append(time.time()-start_time)

        # update sigma vec
        self.sigma_vec.append(self._calculate_prior_sigma())

        # find incumbent
        with warnings.catch_warnings(): 
            warnings.simplefilter(action='ignore',category=gpytorch.utils.warnings.GPInputWarning)
            train_pred = self.model.predict(self.train_x)  
        fmin_index = train_pred.argmin().item()
        self.confs_inc.append(self.train_confs[fmin_index])
        self.f_inc_obs.append(self.train_y[fmin_index].item())
        self.f_inc_est.append(train_pred[fmin_index].item())
    
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
        if self.estimation_method == 'MAP':
            acqobj = LowerConfidenceBound(self.model,kappa=self.kappa)
        else:
            acqobj = LowerConfidenceBoundMCMC(self.model,kappa=self.kappa)
        
        acqopt = AcqFunOptimizer(
            acq_fun=acqobj,
            ndim = len(self.config.quant_index),
            num_starts = max(10,2*len(self.config.quant_index)),
            x0=self.confs_inc[-1].get_array(),
            num_jobs=1 # TODO: add support for parallelization
        )
        start_time = time.time()
        next_x = acqopt.run()
        self.acqopt_time.append(time.time()-start_time)
        self.confs_cand.append(self.config.get_conf_from_array(next_x))
        self.acq_vec.append(acqopt.obj_sign*acqopt.f_inc)
