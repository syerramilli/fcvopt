import numpy as np
import time
import torch
from botorch.acquisition import ExpectedImprovement

from .optimize_acq import _optimize_botorch_acqf
from .fcvopt import FCVOpt
from ..models.gpregression import GPR
from ..models.multitaskgp import MultitaskGPModel, MultiTaskGPConstantCorrModel
from ..fit.mll_scipy import fit_model_scipy

from ..configspace import ConfigurationSpace
from typing import Callable

class SingleTaskExpectedImprovement(ExpectedImprovement):
    def __init__(self,model:MultitaskGPModel,task_idx:int,best_task:float,**kwargs):
        super().__init__(model=model,best_f=best_task,**kwargs)
        self.register_buffer('task_idx',torch.tensor([[task_idx]]).long()) 
    
    def _mean_and_sigma(self,X:torch.Tensor):
        mean,sigma = self.model.predict(X,self.task_idx.repeat(X.shape[-2],1),return_std=True)
        mean = mean.squeeze(-1)
        sigma = sigma.squeeze(-1)
        return mean,sigma


class MTBOCVOpt(FCVOpt):
    def __init__(
        self,
        obj:Callable,
        n_folds:int,
        config:ConfigurationSpace,
        fold_initialization:str='random',
        minimize:bool=True,
        fold_selection_criterion:str='single-task-ei',
        constant_task_corr:bool=False,
        **kwargs
    ):
        super().__init__(
            obj=obj,config=config,
            n_folds=n_folds,n_repeats=1,
            fold_selection_criterion=fold_selection_criterion,
            fold_initialization=fold_initialization,
            minimize=minimize,acq_function=None,
            **kwargs
        )
        self.constant_task_corr = constant_task_corr
    
    def _construct_model(self):
        if self.constant_task_corr:
            return MultiTaskGPConstantCorrModel(
                train_x = (self.train_x,self.train_folds),
                train_y = self.sign_mul*self.train_y,
                num_tasks=self.n_folds
            ).double()

        return MultitaskGPModel(
            train_x = (self.train_x,self.train_folds),
            train_y = self.sign_mul*self.train_y,
            num_tasks=self.n_folds
        ).double()
    
    def _acquisition(self) -> None:
        # acquisition for x is different from that of BayesOpt

        start_time = time.time()
        # first estimate the averaged quantities
        # impute the folds for which there are no obsevations
        fold_idxs = np.arange(self.n_folds)
        #new_train_y = self.train_y.clone()
        # for j,x in enumerate(self.train_x):
        #     x2 = x.clone().unsqueeze(0).repeat(self.n_folds-1,1)
        #     folds_impute = torch.tensor([i for i in fold_idxs if i!=self.train_folds[j]])
        #     with torch.no_grad():
        #         out_means = self.model.predict(x2,folds_impute).sum()
        #     new_train_y[j] = (new_train_y[j]+out_means)/self.n_folds

        # obtain predictions for the average of tasks
        with torch.no_grad():
            new_train_y = self.model.predict(self.train_x)
        
        # fit a new surrogate model for the average of tasks
        new_model = GPR(
            self.train_x,new_train_y
        )
        _ = fit_model_scipy(new_model,num_restarts=5)
        _ = new_model.eval()

        # optimize EI on the average of tasks
        acqobj = ExpectedImprovement(new_model,best_f=new_train_y.max())
        new_x, max_acq = _optimize_botorch_acqf(
            acq_function=acqobj,
            d=self.train_x.shape[-1],
            q=1,
            num_restarts = 20,
            n_jobs=self.n_jobs,
            raw_samples=128
        )
        self.confs_cand.append([self.config.get_conf_from_array(x.numpy()) for x in new_x])
        self.acq_vec.append(max_acq.item())

        # fold acquisition - choose task which has the largest fold-wise EI
        # at the new x
        fold_idxs = np.arange(self.n_folds)

        if self.fold_selection_criterion == 'random':
            self.folds_cand.append(
                np.random.choice(
                    fold_idxs,size=1,replace=True
                ).tolist()
            )
        elif self.fold_selection_criterion == 'single-task-ei':
            # shuffling to prevent ties among folds
            np.random.RandomState(0).shuffle(fold_idxs)

            fold_metrics = []
            for fold_idx in fold_idxs:
                with torch.no_grad():
                    y_fold_idx = self.model.predict(
                        self.train_x,i=torch.tensor([[fold_idx]]).long().repeat(self.train_x.shape[0],1)
                    )
                
                acqobj_fold = SingleTaskExpectedImprovement(self.model,fold_idx,y_fold_idx.max())
                with torch.no_grad():
                    fold_metrics.append(
                        acqobj_fold(new_x).item()
                    )
            
            self.folds_cand.append(
                [fold_idxs[np.argmax(fold_metrics)]]
            )
        
        self.acqopt_time.append(time.time()-start_time)