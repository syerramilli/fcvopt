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
from typing import Callable,List,Union,Tuple,Optional,Dict

class FCVOpt(BayesOpt):
    def __init__(
        self,
        obj:Callable,
        n_folds:int,
        config:ConfigurationSpace,
        correlation_kernel_class:Optional[str]=None,
        kappa:float=2.,
        verbose:int=0.,
        save_iter:Optional[int]=None,
        save_dir:Optional[int]=None
    ):
        super().__init__(
            obj=obj,config=config,correlation_kernel_class=correlation_kernel_class,
            kappa=kappa,verbose=verbose,save_iter=save_iter,save_dir = save_dir
        )
        # fold indicies and candidates not present in BayesOpt
        self.n_folds = n_folds
        self.train_folds = None
        self.folds_cand = []

    def _initialize(self,n_init:Optional[int]=None):
        if self.train_confs is None:
            if n_init is None:
                n_init = len(self.config.quant_index) + 1
            
            self.config.seed(np.random.randint(2e+4))
            self.train_confs = self.config.latinhypercube_sample(n_init)    
            self.train_folds = torch.randint(self.n_folds,size=(n_init,1)).double()

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
        return HGP(
            train_x = (self.train_x,self.train_folds),
            train_y = self.train_y,
            correlation_kernel_class=self.correlation_kernel_class,
            noise=1e-4,
            fix_noise=False
        ).double()
    
    def _acquisition(self) -> None:
        # acquisition for x is the same as BayesOpt
        super()._acquisition()

        # fold acquisition - random for now
        # TODO: implement the more systemic criterion for fold selection
        self.folds_cand.append(np.random.choice(self.n_folds))