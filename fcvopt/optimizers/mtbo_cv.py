import numpy as np
import os 
import time
import torch
from .bayes_opt import BayesOpt
from ..models import GPR
from ..models.multitaskgp import MultitaskGPModel
from ..models.fit_model import fit_model_unconstrained
from ..acquisition import ExpectedImprovement
from .acqfunoptimizer import AcqFunOptimizer

from ..configspace import ConfigurationSpace
from ..util.samplers import stratified_sample
from typing import Callable,List,Union,Tuple,Optional,Dict

class MTBOCVOpt(BayesOpt):
    def __init__(
        self,
        obj:Callable,
        n_folds:int,
        config:ConfigurationSpace,
        estimation_method:str='MAP',
        fold_selection_criterion:str='single-task-ei',
        correlation_kernel_class:Optional[str]=None,
        kappa:float=2.,
        verbose:int=0.,
        save_iter:Optional[int]=None,
        save_dir:Optional[int]=None
    ):
        super().__init__(
            obj=obj,config=config,
            estimation_method=estimation_method,
            correlation_kernel_class=correlation_kernel_class,
            kappa=kappa,verbose=verbose,save_iter=save_iter,save_dir = save_dir
        )
        # fold indices and candidates not present in BayesOpt
        # TODO: add checks for the validity of fold_selection criterion
        self.fold_selection_criterion = fold_selection_criterion
        self.n_folds = n_folds
        self.train_folds = None
        self.folds_cand = []

    def _initialize(self,n_init:Optional[int]=None):
        if self.train_confs is None:
            n_init = 0 if n_init is None else n_init
            # need at least one observation per fold to estimate the correlation
            # parameters. Also need atleast D+1 observations to kickstart
            # decent estimates
            n_init = max(n_init,len(self.config.quant_index) + 1,self.n_folds)
            
            self.config.seed(np.random.randint(2e+4))
            self.train_confs = self.config.latinhypercube_sample(n_init)
            # apply stratified sampling to the folds
            self.train_folds = torch.from_numpy(stratified_sample(self.n_folds,n_init)).long()

            # maintain list of fold-wise observed minimas
            self.fold_wise_opts = [np.inf]*self.n_folds

            for conf,fold_idx in zip(self.train_confs,self.train_folds):
                x,y,eval_time = self._evaluate(conf,fold_idxs=[fold_idx.item()])
                self.train_x.append(x)
                self.train_y.append(y)
                self.obj_eval_time.append(eval_time)

                # update folds' minima when eligible
                if y < self.fold_wise_opts[fold_idx.item()]:
                    self.fold_wise_opts[fold_idx.item()] = y
                
            
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
            self.train_folds = torch.cat([self.train_folds,torch.tensor(next_fold).to(self.train_folds)])
            self.obj_eval_time.append(eval_time)

            # update fold-widse minimas when eligible
            if next_y < self.fold_wise_opts[next_fold[0]]:
                self.fold_wise_opts[next_fold[0]] = next_y
    
    def _construct_model(self):
        return MultitaskGPModel(
            train_x = (self.train_x,self.train_folds),
            train_y = self.train_y,
            num_tasks=self.n_folds,
            correlation_kernel_class=self.correlation_kernel_class,
            noise=1e-4,
        ).double()
    
    def _acquisition(self) -> None:
        # acquisition for x is different from that of BayesOpt

        start_time = time.time()
        # first estimate the averaged quantities
        # impute the folds for which there are no obsevations
        fold_idxs = np.arange(self.n_folds)
        new_train_y = self.train_y.clone()
        for j,x in enumerate(self.train_x):
            x2 = x.clone().unsqueeze(0).repeat(self.n_folds-1,1)
            folds_impute = torch.tensor([i for i in fold_idxs if i!=self.train_folds[j]])
            with torch.no_grad():
                out_means = self.model.predict(x2,folds_impute).sum()
            new_train_y[j] = (new_train_y[j]+out_means)/self.n_folds
        
        # optimize EI on the single task GP
        acqobj = ExpectedImprovement(self.model,f_best=new_train_y.min())
        acqopt = AcqFunOptimizer(
            acq_fun=acqobj,
            ndim = len(self.config.quant_index),
            num_starts = max(10,2*len(self.config.quant_index)),
            x0=self.confs_inc[-1].get_array(),
            num_jobs=1 # TODO: add support for parallelization
        )
        next_x = acqopt.run()
        self.confs_cand.append(self.config.get_conf_from_array(next_x))
        self.acq_vec.append(acqopt.obj_sign*acqopt.f_inc)

        # Given new x choose the next fold

        # fold acquisition - chose fold which has the highest fold-wise EI
        # at the new x
        if self.fold_selection_criterion == 'random':
            self.folds_cand.append(np.random.choice(self.n_folds))
        elif self.fold_selection_criterion == 'single-task-ei':
            fold_idxs = np.arange(self.n_folds)

            # shuffling to prevent ties among folds
            np.random.shuffle(fold_idxs)

            next_x = torch.tensor(
                self.confs_cand[-1].get_array()
            ).to(self.train_x).reshape(1,-1)
            
            with torch.no_grad():
                fold_metrics = [
                    ExpectedImprovement(self.model,self.fold_wise_opts[fold_idx])(
                        next_x,i=torch.tensor([fold_idx])
                    ).item() for fold_idx in fold_idxs
                ]

            self.folds_cand.append(
                fold_idxs[np.argmax(fold_metrics)]
            )
        
        self.acqopt_time.append(time.time()-start_time)
    
    def _calculate_prior_sigma(self) -> float:
        return 0