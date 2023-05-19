import numpy as np
import os 
import time
import torch
from .fcvopt import FCVOpt
from ..models import GPR

from ..configspace import ConfigurationSpace
from ..util.samplers import stratified_sample
from typing import Callable,List,Union,Tuple,Optional,Dict

class CVRandOpt(FCVOpt):
    def __init__(
        self,
        obj:Callable,
        config:ConfigurationSpace,
        n_folds:int,
        n_repeats:int=1,
        fold_initialization:str='random',
        minimize:bool=True,
        acq_function:str='LCB',
        **kwargs
    ):
        if acq_function == 'EI':
            raise RuntimeError('Expected improvment not implemented for CVRandOpt')

        super().__init__(
            obj=obj,config=config,
            n_folds=n_folds,n_repeats=n_repeats,
            fold_selection_criterion='random', # no model
            fold_initialization=fold_initialization,
            minimize=minimize,acq_function=acq_function,
            **kwargs
        )

    def _construct_model(self):
        return GPR(
            train_x=self.train_x,
            train_y=self.sign_mul*self.train_y
        )