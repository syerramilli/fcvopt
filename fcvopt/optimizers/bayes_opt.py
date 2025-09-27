import numpy as np
import os 
import time
import torch
import gpytorch
import warnings
import joblib

from .optimize_acq import _optimize_botorch_acqf
from ..models import GPR
from ..fit.mll_scipy import fit_model_scipy
from ..configspace import ConfigurationSpace
from ConfigSpace import Configuration

from botorch.acquisition import (
    ExpectedImprovement,qExpectedImprovement,
    UpperConfidenceBound, qUpperConfidenceBound,
    qKnowledgeGradient
)
from botorch.sampling import SobolQMCNormalSampler

from typing import Callable, List, Optional, Dict, Tuple
from collections import OrderedDict

class BayesOpt:
    """Bayesian Optimization for optimizing a given objective function.

    This class implements a Bayesian optimization loop that iteratively fits a Gaussian 
    Process model to the objective function and prpopses new configurations to evaulate 
    via an acquisition function.

    Args:
        obj: The objective function mapping a configuration dict to a scalar value.
        config: The search space.
        minimize: If True, minimizes the objective; otherwise maximizes it. Defaults to True.
        acq_function: Acquisition function to use. One of {'EI', 'LCB', 'KG'}. Defaults to 'EI'.
        acq_function_options: Additional keyword arguments passed to the acquisition function constructor. 
                Defaults to None.
        batch_acquisition: If True, a batch of configurations (the number specifed by `acquisition_q`) 
                is selected for each iteration . Defaults to False.
        acquisition_q (int, optional):
            Number of points in each proposed batch when `batch_acquisition` is True. Defaults to 1.
        verbose: Verbosity level to print to console; 0=no output, 1=summary at end, 2=detailed per-iteration 
                log. Defaults to 1.
        save_iter: Interval (in iterations) at which to auto-save state. Defaults to None.
        save_dir: Directory path in which to save state files if `save_iter` is set. Defaults to None.
        n_jobs: Number of parallel jobs for objective evaluation and model fitting. Use -1 to utilize all 
                available CPU cores. Defaults to 1.
    """
    def __init__(
        self,
        obj:Callable,
        config:ConfigurationSpace,
        minimize:bool=True,
        acq_function:str='EI',
        acq_function_options:Optional[Dict]={},
        batch_acquisition:bool=False,
        acquisition_q:int=1,
        verbose:int=1,
        save_iter:Optional[int]=None,
        save_dir:Optional[int]=None,
        n_jobs:int=1
    ):
        self.obj = obj
        self.config = config
        self.minimize=minimize
        self.sign_mul = -1 if self.minimize else 1
        self.acq_function = acq_function
        self.batch_acquisition = batch_acquisition
        self.acquisition_q = acquisition_q

        self.acq_function_options = acq_function_options
        self.verbose=verbose
        self.save_iter = save_iter
        self.save_dir = save_dir
        if not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
            raise ValueError(f"n_jobs must be -1 (all cores) or a positive integer; got {n_jobs!r}")
        self.n_jobs = n_jobs

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
        self.fit_time = []
        self.acqopt_time = []
        self.obj_eval_time = []
        self.initial_params = None
    
    def run(self,n_iter:int,n_init:Optional[int]=None) -> Dict:
        """Run the Bayesian optimization loop for a specified number of iterations.

        This method will perform `n_iter` iterations of model fitting, acquisition optimization, 
        and objective evaluation. The total number of configurations evaluated will be
        `n_init + n_iter - 1`, where `n_init` is the number of random initial points.

        Args:
            n_iter: Number of Bayesian optimization iterations to perform.
            n_init: Number of random initial points. If None, defaults to `len(config.quant_index) + 1`.

        Returns:
            dict: An ordered dictionary containing:
                - 'conf_inc': Best configuration found.
                - 'f_inc_obs': Observed objective value at best config.
                - 'f_inc_est': Model’s estimate at best config.
        """
        # TODO: change n_iter to n_trials for more user-friendly API
        output_header = '%6s %10s %10s %10s' % \
                    ('iter', 'f_inc_obs', 'f_inc_est','acq_cand')
        for i in range(n_iter):
            # either initialize observations or evaluate the next candidate(s)
            self._initialize(n_init)

            # fit model and find incumbent
            self._fit_model_and_find_inc(i)
        
            # acquisition find next candidate
            self._acquisition()

            if self.save_iter is not None:
                if i % self.save_iter == 0:
                    self.save_to_file(self.save_dir)

            # update verbose statements
            if self.verbose >= 2:
                if i%10 == 0:
                    # print header every 10 iterations
                    print(output_header)
                print('%6i %10.3e %10.3e %10.3e' %\
                      (i, self.f_inc_obs[-1],self.f_inc_est[-1],
                      self.acq_vec[-1]))
        
        if self.verbose >= 1:
            print('')
            print('Number of candidates evaluated.....: %g' % len(self.train_confs))
            print('Observed obj at incumbent..........: %g' % self.f_inc_obs[-1])
            print('Estimated obj at incumbent.........: %g' % self.f_inc_est[-1])
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
    
    def optimize(self, n_trials:int, n_init:Optional[int]=None) -> Dict:
        """Run Bayesian optimization for a fixed **total** number of trials.

        Unlike :meth:`run`, which performs exactly ``n_iter`` acquisition steps,
        this method targets a total trial budget (initial random evaluations
        **plus** acquired candidates). It computes the number of acquisitions as:

            acquisitions = n_trials - n_init + 1

        where ``n_init`` is the number of initial random configurations evaluated
        before starting Bayesian optimization. The loop then proceeds to perform
        that many acquisitions.

        Args:
            n_trials (int):
                Total number of objective evaluations desired, including the initial
                random evaluations and all subsequent acquisitions. Must be ≥ 1.
            n_init (int, optional):
                Number of initial random configurations to evaluate before BO begins.
                If ``None``, defaults to ``len(self.config.quant_index) + 1``.
                Must satisfy ``1 ≤ n_init ≤ n_trials``.
                (If ``n_init == n_trials``, exactly **one** acquisition is performed.)

        Returns:
            Dict:
                Ordered results identical to :meth:`run`, with keys:
                - ``'conf_inc'``: incumbent configuration,
                - ``'f_inc_obs'``: observed objective at incumbent,
                - ``'f_inc_est'``: model-estimated objective at incumbent.

        Raises:
            ValueError: If ``n_init`` is not in ``[1, n_trials]``.

        Examples:
            >>> # Target exactly 40 total evaluations with 8 random starts
            >>> results = bo.optimize(n_trials=40, n_init=8)
            >>> results['f_inc_obs']
        """
        # resolve n_init
        if n_init is None:
            n_init = len(self.config.quant_index) + 1

        if not (1 <= n_init <= n_trials):
            raise ValueError(f"n_init must be in [1, {n_trials}], got {n_init!r}")

        # compute how many acquisitions to perform after initialization
        n_iter = n_trials - n_init + 1
        return self.run(n_iter=n_iter, n_init=n_init)

    @property
    def stats_keys(self) -> List:
        return [
            'f_inc_obs','f_inc_est','acq_vec',
            'confs_inc','confs_cand',
            'fit_time','acqopt_time','obj_eval_time',
        ]

    @property
    def data_keys(self)->List:
        return ['train_x','train_y']

    def save_to_file(self, path:str):
        """Save optimization stats, observations, and model state.

        Args:
            path: Directory in which to save the data.
        """ 
        #  optimization statistics
        stats = {
            key:getattr(self,key) for key in self.stats_keys
        }
        joblib.dump(stats,os.path.join(path,'stats.pkl'))
        # Observations
        _ = torch.save({
            key:getattr(self,key) for key in self.data_keys
        },os.path.join(path,'model_train.pt'))
        # model state dict
        _ = torch.save(self.model.state_dict(),os.path.join(path,'model_state.pth'))


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
    
    def _evaluate(self, conf:Configuration, **kwargs) -> Tuple[np.ndarray, float, float]:
        """Evaluate the objective on a single configuration.

        Args:
            conf: A single configuration object.
            **kwargs: Additional keyword args passed to `self.obj`.

        Returns:
            tuple: (ndarray, float, float) where the first element is the
                numeric array from `conf.get_array()`, the second is the
                objective value, and the third is elapsed time.
        """
        start_time = time.time()
        y = self.obj(dict(conf),**kwargs)
        eval_time = time.time()-start_time
        return conf.get_array(),y,eval_time

    def _evaluate_confs(self, confs_list:List[Configuration],**kwargs) -> List[Tuple[np.ndarray, float, float]]:
        """Evaluate multiple configurations, optionally in parallel.

        Args:
            confs_list: List of configurations to evaulate the objective on.
            **kwargs: Additional keyword args passed to `self.obj`.

        Returns:
            List[tuple]: Each element is the return of `_evaluate`.
        """
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
    
    def _fit_model_and_find_inc(self, i:int) -> None:
        # construct model
        self.model = self._construct_model()

        start_time = time.time()
        if self.initial_params is not None:
            self.model.initialize(**self.initial_params)

        _ = fit_model_scipy(model = self.model,num_restarts = 5, n_jobs=self.n_jobs, rng_seed=i)
            
        self.fit_time.append(time.time()-start_time)

        # disable model gradients
        self.initial_params = OrderedDict()
        for name,parameter in self.model.named_parameters():
            parameter.requires_grad_(False)
            self.initial_params[name] = parameter

        # find incumbent
        with warnings.catch_warnings(),torch.no_grad(): 
            warnings.simplefilter(action='ignore',category=gpytorch.utils.warnings.GPInputWarning)
            train_pred = self.model.predict(self.train_x)

        fmin_index = train_pred.argmax().item() 
        self.confs_inc.append(self.train_confs[fmin_index])
        self.f_inc_obs.append(self.train_y[fmin_index].item())
        self.f_inc_est.append(self.sign_mul*train_pred[fmin_index].item())
    
    def _construct_model(self) -> GPR:
        return GPR(
            train_x = self.train_x,
            train_y = self.sign_mul*self.train_y,
        ).double()
    
    def _acquisition(self) -> None:
        if self.acq_function == 'EI':
            best_f = -self.f_inc_est[-1] if self.minimize else self.f_inc_est[-1]
            if self.batch_acquisition:
                acqobj = qExpectedImprovement(self.model,best_f, sampler = SobolQMCNormalSampler(128, seed=0))
            else:
                acqobj = ExpectedImprovement(self.model, best_f)
        elif self.acq_function == 'LCB':
            if self.batch_acquisition:
                acqobj = qUpperConfidenceBound(self.model, torch.tensor(4.), sampler = SobolQMCNormalSampler(128, seed=0))
            else:
                acqobj = UpperConfidenceBound(self.model, torch.tensor(4.))
        elif self.acq_function == 'KG':
            num_fantasies = 32
            acqobj = qKnowledgeGradient(
                self.model, sampler=SobolQMCNormalSampler(num_fantasies, seed=0), 
                num_fantasies=num_fantasies
            )

        start_time = time.time()
        new_x, max_acq = _optimize_botorch_acqf(
            acq_function=acqobj,
            d=self.train_x.shape[-1],
            q=1 if not self.batch_acquisition else self.acquisition_q,
            num_restarts = 10 if self.acq_function == 'KG' else 20,
            n_jobs=self.n_jobs,
            raw_samples=128
        )

        end_time = time.time()
        self.acqopt_time.append(end_time-start_time)
        self.confs_cand.append([self.config.get_conf_from_array(x.numpy()) for x in new_x])
        self.acq_vec.append(-max_acq.item() if self.acq_function =='LCB' else max_acq.item())