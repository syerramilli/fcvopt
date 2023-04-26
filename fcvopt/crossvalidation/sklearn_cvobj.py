import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from joblib import Parallel,delayed
from typing import Callable,List,Optional,Dict,Tuple

from ..crossvalidation.cvobjective import CVObjective

class SklearnCVObj(CVObjective):
    def __init__(
        self,
        estimator,
        X, y, 
        loss_metric,
        needs_proba:bool=False, 
        n_splits=10, 
        n_repeats=1,
        holdout=False,
        task='regression', 
        scale_output=False, 
        input_preprocessor=None,
        stratified=False, 
        num_jobs=1
    ):
        super().__init__(
            X, y, loss_metric, n_splits=n_splits, n_repeats=n_repeats, 
            holdout=holdout, task=task, scale_output=scale_output, 
            input_preprocessor=input_preprocessor, 
            stratified=stratified,num_jobs=num_jobs
        )

        self.estimator = estimator
        self.needs_proba = needs_proba
    
    def construct_model(self,params):
        return clone(self.estimator).set_params(**params)
    
    def _fit_and_test(self, params, train_index, test_index):
        model = self.construct_model(params)

        # fold train and test sets
        if isinstance(self.X, pd.DataFrame):
            X_train = self.X.iloc[train_index,:];X_test = self.X.iloc[test_index,:]
        else:
            X_train = self.X[train_index,...];X_test = self.X[test_index,...]

        if self.input_preprocessor is not None:
            input_preprocessor = clone(self.input_preprocessor).fit(self.X[train_index,...])
            X_train = input_preprocessor.transform(X_train)
            X_test = input_preprocessor.transform(X_test)

        scorer = make_scorer(self.loss_metric,needs_proba=self.needs_proba)

        # fit model
        model.fit(X_train,self.y[train_index,...])

        return scorer(model,X_test,self.y[test_index,...])


class XGBoostCVObjEarlyStopping(SklearnCVObj):
    def __init__(self, 
        early_stopping_rounds:int,validation_split:float=0.1,
        **kwargs):
        super().__init__(**kwargs)
        self.early_stopping_rounds=early_stopping_rounds
        self.validation_split = validation_split

    def _fit_and_test(self, params, train_index, test_index):
        model = self.construct_model(params).set_params(**{'early_stopping_rounds':self.early_stopping_rounds})
        scorer = make_scorer(self.loss_metric,needs_proba=self.needs_proba)
        # fold train and test sets
        if isinstance(self.X, pd.DataFrame):
            X_train = self.X.iloc[train_index,:];X_test = self.X.iloc[test_index,:]
        else:
            X_train = self.X[train_index,...];X_test = self.X[test_index,...]

        if self.input_preprocessor is not None:
            input_preprocessor = clone(self.input_preprocessor).fit(self.X[train_index,...])
            X_train = input_preprocessor.transform(X_train)
            X_test = input_preprocessor.transform(X_test)

        # for early stopping split the train set further into training and validation
        # the performance on the validation split will be used as early stopping criterion
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,self.y[train_index,...],test_size=self.validation_split,
            stratify=self.y[train_index,...] if self.stratified and 'classification' in self.task else None
        )

        # fit model
        model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=0)

        return scorer(model,X_test,self.y[test_index,...])