import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold,RepeatedStratifiedKFold
from joblib import Parallel,delayed
from typing import Callable,List,Optional,Dict,Tuple, Union

class CVObjective:
    '''Base class for cross-validation objective functions

    Args:
        X: input data with dimensions (N x D)
        y: target data with dimensions (N x 1)
        task: task type. Must be one of 'regression', 'binary_classification', or
            'multiclass_classification'. If the task is classification, `y` is internally
            encoded using `sklearn.preprocessing.LabelEncoder`.
        loss_metric: loss metric to minimize. Must be a function/callable that takes
            in two arguments: y_true and y_pred and returns a scalar loss value. 
        n_splits: number of splits for K-fold CV (default: 5)
        n_repeats: number of repeats for K-fold CV (default: 5)
        holdout: whether to perform holdout CV. If True, only the first
            train-test split is used for evaluation. (default: False)
        scale_output: whether to standardize the output for a regression task. Relevant
            only when `task` is 'regression'. (default: False)
        input_preprocessor: A scikit-learning transformer object that is used to preprocess
            the input data. If not None, the transformer is learned separately for each
            train-test split. (default: None)
        stratified: whether to perform stratified CV for classification tasks. Relevant only
            when `task` is 'binary_classification' or 'multiclass_classification'. (default: True)
        num_jobs: number of jobs to run in parallel (default: 1)
    '''
    def __init__(
        self,
        X:Union[np.ndarray,pd.DataFrame],
        y:Union[np.ndarray,pd.Series],
        task:str,
        loss_metric:Callable,
        n_splits:int=5,
        n_repeats:int=5,
        holdout:bool=False,
        scale_output:bool=False,
        input_preprocessor=None,
        stratified:bool=True,
        num_jobs:int=1 # writing it is as num_jobs to distinguish it from sklearn's n_jobs
    ):
        self.X = X
        self.y = y
        self.task = task
        self.loss_metric = loss_metric
        self.stratified = stratified
        if self.stratified and 'classification' in self.task:
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
        self.input_preprocessor = input_preprocessor
        self.num_jobs = num_jobs
    
    def construct_model(self, params: Dict,**kwargs):
        '''Constructs and returns a model given the hyperparameters `params`

        Args:
            params: a dictionary of hyperparameters
            kwargs: additional keyword arguments
        '''
        pass

    def _fit_and_test(self,params,train_index,test_index):
        pass

    def cvloss(
        self,
        params:Dict,
        fold_idxs:Optional[List[int]]=None,
        all:bool=False
    ):
        '''
        Computes the cross-validation loss for the given hyperparameters `params` at 
        the given fold indices `fold_idxs`.

        Args:
            params: a dictionary of hyperparameters
            fold_idxs: a list of fold indices to use for computing the CV loss. If None,
                all folds are used if `self.holdout` is False, otherwise only the first
                fold is used. (default: None)
            all: whether to return the loss values for all folds. If False, the mean loss
                across folds is returned. (default: False)
        '''
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
        