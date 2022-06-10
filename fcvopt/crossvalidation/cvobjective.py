import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold,RepeatedStratifiedKFold
from joblib import Parallel,delayed
from typing import Callable,List,Optional,Dict,Tuple

class CVObjective:
    def __init__(
        self,
        X,y,
        loss_metric:Callable,
        n_splits:int=5,
        n_repeats:int=5,
        holdout:bool=False,
        task:str='regression',
        scale_output:bool=False,
        input_preprocessor=None,
        stratified:bool=True,
        num_jobs:int=1 # writing it is as num_jobs to distinguish it from sklearn's n_jobs
    ):
        self.X = X
        self.y = y
        self.task = task
        self.loss_metric = loss_metric
        if stratified:
            self.cv = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats)
        else:
            self.cv = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats)
        self.train_test_splits = list(self.cv.split(X,y)) # initial splits
        self.holdout = holdout
        if self.holdout:
            self.train_test_splits = self.train_test_splits[0:1]

        if 'classification' in self.task:
            # encoded categorical outputs
            self.y = LabelEncoder().fit_transform(self.y)
        self.scale_output = scale_output
        if self.scale_output and self.task=='regression':
            self.y_std = np.std(self.y)
            self.y_mean = np.mean(self.y)
            self.y = (self.y-self.y_mean)/self.y_std

        self.input_preprocessor = input_preprocessor
        self.num_jobs = num_jobs
    
    def construct_model(self,params,**kwargs):
        pass

    def _fit_and_test(self,params,train_index,test_index):
        pass

    def cvloss(self,params:Dict,fold_idxs:Optional[List[int]]=None,all:bool=False):
        if fold_idxs is None:
            if self.holdout:
                fold_idxs = [0]
            else:
                # default: perform full replicated K-fold CV
                fold_idxs = np.arange(self.cv.get_n_splits())

        fold_losses = Parallel(n_jobs=self.num_jobs)(
            delayed(self._fit_and_test)(
                params,self.train_test_splits[idx][0],self.train_test_splits[idx][1]
            ) for idx in fold_idxs
        )

        if all:
            return np.array(fold_losses)
        
        return np.mean(fold_losses)
        