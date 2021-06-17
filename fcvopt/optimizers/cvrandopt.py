import numpy as np
import os 
import time
import torch
from .bayes_opt import BayesOpt
from ..models import GPR
from ..models.mcmc_utils import mcmc_run
from ..acquisition import LowerConfidenceBoundMCMC
from .acqfunoptimizer import AcqFunOptimizer

from ..configspace import ConfigurationSpace
from ..util.samplers import stratified_sample
from typing import Callable,List,Union,Tuple,Optional,Dict

class CVRandOpt(BayesOpt):
    def __init__(
        self,
        obj:Callable,
        n_folds:int,
        n_repeats:int,
        config:ConfigurationSpace,
        deterministic:str=False,
        estimation_method:str='MAP',
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
    
    def _acquisition(self) -> None:
        # acquisition for x is the same as BayesOpt
        super()._acquisition()
        
        total_num_folds = self.n_folds
        if self.n_repeats > 1 and self.train_folds.flatten().unique().shape[0] < self.n_folds:
            # consider sampling from other replicates only if all folds of the 
            # first replicate have been evaluated at least once
            total_num_folds = self.n_folds*self.n_repeats

        fold_idxs = np.arange(total_num_folds)
        # randomly select folds
        self.folds_cand.append(np.random.choice(fold_idxs))