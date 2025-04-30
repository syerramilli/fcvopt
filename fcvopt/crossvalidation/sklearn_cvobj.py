import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import Parallel,delayed
from typing import Callable,List,Optional,Dict,Tuple, Union

from ..crossvalidation.cvobjective import CVObjective

class SklearnCVObj(CVObjective):
    '''Cross-validation objective function for scikit-learn models

    Args:
        X: input data with dimensions (N x D)
        y: target data with dimensions (N x 1)
        estimator: A scikit-learn estimator object whose hyperparameters are to be optimized.
            on the given data. If this is not a standard scikit-learn estimator, it must inherit
            from `sklearn.base.BaseEstimator` and implement `fit` and `predict` (and `predict_proba` 
            if `needs_proba` is True) methods.
        task: task type. Must be one of 'regression', 'binary_classification', or
            'multiclass_classification'. If the task is classification, `y` is internally
            encoded using `sklearn.preprocessing.LabelEncoder`.
        loss_metric: loss metric to minimize. Must be a function/callable that takes
            in two arguments: y_true and y_pred and returns a scalar loss value.
        needs_proba: whether the loss metric needs class probabilities. Relevant only
            when `task` is 'binary_classification' or 'multiclass_classification'.
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
        estimator: BaseEstimator,
        X:Union[np.ndarray,pd.DataFrame], 
        y:Union[np.ndarray,pd.Series],
        task:str,
        loss_metric:Callable,
        needs_proba:bool=False, 
        n_splits=10, 
        n_repeats=1,
        holdout=False,
        scale_output=False, 
        input_preprocessor=None,
        stratified=False, 
        num_jobs=1,
        rng_seed=None
    ):
        super().__init__(
            X, y, task, loss_metric, n_splits=n_splits, n_repeats=n_repeats, 
            holdout=holdout, scale_output=scale_output, 
            input_preprocessor=input_preprocessor, 
            stratified=stratified,num_jobs=num_jobs
        )

        self.estimator = estimator
        self.needs_proba = needs_proba
        self._rng = np.random.default_rng(rng_seed)
    
    def construct_model(self, params:Dict) -> BaseEstimator:
        model = clone(self.estimator).set_params(**params)
        # if the estimator is stochastic give it a reproducible seed
        if hasattr(model,'random_state'):
            seed = self._rng.integers(0, np.iinfo(np.int32).max)
            model.set_params(**{'random_state':seed})
        
        return model
    
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

        y_train = self.y[train_index,...]; y_test = self.y[test_index,...]
        if self.scale_output and self.task=='regression':
            y_std = np.std(self.y[train_index,...])
            y_mean = np.mean(self.y[train_index,...])
            y_train = (y_train-y_mean)/y_std
            y_test = (y_test-y_mean)/y_std

        scorer = make_scorer(self.loss_metric,needs_proba=self.needs_proba)
        # fit model
        model.fit(X_train,y_train)

        return scorer(model,X_test,y_test)


class XGBoostCVObjEarlyStopping(SklearnCVObj):
    '''Cross-validation objective function for XGBoost models with early stopping

    This uses the scikit-learn API of XGBoost. To implement early stopping, the training
    set within each train-test split is further split into a training and validation set 
    (with stratification if `stratified` is True and the taskis classification). The loss 
    metric on the validation set is used as the early stopping criterion.

    Args:
        early_stopping_rounds: number of rounds to wait for the loss to improve before
            stopping the training
        validation_split: fraction of the training set to use as the validation set for
            early stopping (default: 0.1)
        **kwargs: additional keyword arguments to `SklearnCVObj`. Note that the `estimator`
            argument must be  `xgboost.XGBRegressor` or a `xgboost.XGBClassifier` object.
    '''
    def __init__(self, 
        early_stopping_rounds:int,
        validation_split:float=0.1,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.early_stopping_rounds=early_stopping_rounds
        self.validation_split = validation_split

    def _fit_and_test(self, params, train_index, test_index):
        model = self.construct_model(params).set_params(
            **{'early_stopping_rounds':self.early_stopping_rounds}
        )

        y_train = self.y[train_index,...]; y_test = self.y[test_index,...]
        if self.scale_output and self.task=='regression':
            y_std = np.std(self.y[train_index,...])
            y_mean = np.mean(self.y[train_index,...])
            y_train = (y_train-y_mean)/y_std
            y_test = (y_test-y_mean)/y_std

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

        if self.task == 'regression':
            _  = model.set_params(**{'base_score':y_train.mean()})

        # for early stopping split the train set further into training and validation
        # the performance on the validation split will be used as early stopping criterion
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,y_train,test_size=self.validation_split,
            stratify=y_train if self.stratified and 'classification' in self.task else None
        )

        # fit model
        model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=0)

        return scorer(model,X_test,y_test)