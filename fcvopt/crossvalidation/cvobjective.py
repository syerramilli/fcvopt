import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold
from joblib import Parallel,delayed
from typing import Callable,List,Optional,Dict,Tuple

class CVObjective:
    def __init__(
        self,
        X,y,
        loss_metric:Callable,
        n_splits:int=5,
        n_repeats:int=5,
        task:str='regression',
        scale_output:bool=False,
        input_preprocessor=None,
        num_jobs:int=1 # writing it is as num_jobs to distinguish it from sklearn's n_jobs
    ):
        self.X = X
        self.y = y
        self.task = task
        self.loss_metric = loss_metric
        self.cv = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats)
        self.train_test_splits = list(self.cv.split(X)) # initial splits

        if 'classification' in self.task:
            # encoded categorical outputs
            self.y = LabelEncoder().fit_transform(self.y)
        self.scale_output = scale_output
        self.input_preprocessor = input_preprocessor
        self.num_jobs = num_jobs
    
    def _fit_and_test(self,params,train_index,test_index):
        pass

    def cvloss(self,params:Dict,fold_idxs:List[int]=[0],all:bool=False):
        fold_losses = Parallel(n_jobs=self.num_jobs)(
            delayed(self._fit_and_test)(
                params,self.train_test_splits[idx][0],self.train_test_splits[idx][1]
            ) for idx in fold_idxs
        )

        if all:
            return np.array(fold_losses)
        
        return np.mean(fold_losses)
        