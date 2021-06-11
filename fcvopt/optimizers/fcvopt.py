import numpy as np
import os 
import time
import torch
from .bayes_opt import BayesOpt
from ..models import HGP
from ..models.mcmc_utils import mcmc_run
from ..acquisition import LowerConfidenceBoundMCMC
from .acqfunoptimizer import AcqFunOptimizer

from ..configspace import ConfigurationSpace
from ..util.samplers import stratified_sample
from typing import Callable,List,Union,Tuple,Optional,Dict

class FCVOpt(BayesOpt):
    def __init__(
        self,
        obj:Callable,
        n_folds:int,
        n_repeats:int,
        config:ConfigurationSpace,
        deterministic:str=False,
        estimation_method:str='MAP',
        fold_selection_criterion:str='variance_reduction',
        fold_initialization:str='random',
        correlation_kernel_class:Optional[str]=None,
        kappa:float=2.,
        verbose:int=0.,
        save_iter:Optional[int]=None,
        save_dir:Optional[int]=None
    ):
        super().__init__(
            obj=obj,config=config,deterministic=deterministic,
            estimation_method=estimation_method,
            correlation_kernel_class=correlation_kernel_class,
            kappa=kappa,verbose=verbose,save_iter=save_iter,save_dir = save_dir
        )
        # fold indices and candidates not present in BayesOpt
        # TODO: add checks for the validity of fold_selection criterion
        self.fold_selection_criterion = fold_selection_criterion
        self.fold_initialization = fold_initialization
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.train_folds = None
        self.folds_cand = []

    def _initialize(self,n_init:Optional[int]=None):
        if self.train_confs is None:
            if n_init is None:
                n_init = len(self.config.quant_index) + 1
            
            self.config.seed(np.random.randint(2e+4))
            self.train_confs = self.config.latinhypercube_sample(n_init)
            # initialize fold to be from among the first replicate    
            #self.train_folds = torch.randint(self.n_folds,(1,1)).repeat(n_init,1).double()
            if self.fold_initialization == 'random':
                self.train_folds = torch.randint(self.n_folds,(n_init,1)).double()
            elif self.fold_initialization == 'stratified':
                self.train_folds = torch.from_numpy(stratified_sample(self.n_folds,n_init)).double().view(-1,1)

            for conf,fold_idx in zip(self.train_confs,self.train_folds):
                x,y,eval_time = self._evaluate(conf,fold_idxs=fold_idx.int().tolist())
                self.train_x.append(x)
                self.train_y.append(y)
                self.obj_eval_time.append(eval_time)
            
            self.train_x = torch.tensor(self.train_x).double()
            self.train_y = torch.tensor(self.train_y).double()
        else:
            # algorithm has been run previously
            # evaluate the next candidate 
            next_conf = self.confs_cand[-1]
            next_fold = [self.folds_cand[-1]]
            next_x,next_y,eval_time = self._evaluate(next_conf,fold_idxs=next_fold)
            self.train_confs.append(next_conf)
            self.train_y = torch.cat([self.train_y,torch.tensor([next_y]).to(self.train_y)])
            self.train_x = torch.cat([self.train_x,torch.tensor(next_x).to(self.train_x).reshape(1,-1)])
            self.train_folds = torch.cat([self.train_folds,torch.tensor(next_fold).to(self.train_folds).reshape(1,-1)])
            self.obj_eval_time.append(eval_time)
    
    def _construct_model(self):
        noise = 1e-4 if self.deterministic else 1e-2
        return HGP(
            train_x = (self.train_x,self.train_folds),
            train_y = self.train_y,
            correlation_kernel_class=self.correlation_kernel_class,
            noise=noise,
            fix_noise=self.deterministic,
            estimation_method=self.estimation_method
        ).double()
    
    def _acquisition(self) -> None:
        # acquisition for x is the same as BayesOpt
        super()._acquisition()
        
        total_num_folds = self.n_folds
        if self.n_repeats > 1 and self.train_folds.flatten().unique().shape[0] < self.n_folds:
            # consider sampling from other replicates only if all folds of the 
            # first replicate have been evaluated at least once
            total_num_folds = self.n_folds*self.n_repeats

        fold_idxs = np.arange(total_num_folds)
        # shuffling to prevent ties among folds
        np.random.shuffle(fold_idxs)

        # fold acquisition
        if self.fold_selection_criterion == 'random':
            self.folds_cand.append(np.random.choice(fold_idxs))
        elif self.fold_selection_criterion == 'variance_reduction':
            next_x = torch.tensor(
                self.confs_cand[-1].get_array()
            ).to(self.train_x).reshape(1,-1)
            
            fold_metrics = self.model._fold_selection_metric(next_x,fold_idxs)
            self.folds_cand.append(
                fold_idxs[np.argmin(fold_metrics)]
            )