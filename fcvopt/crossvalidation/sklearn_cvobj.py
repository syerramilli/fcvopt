import numpy as np
from sklearn.base import clone
from sklearn.metrics import make_scorer
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
        n_splits=5, 
        n_repeats=5,
        holdout=False,
        task='regression', 
        scale_output=False, 
        input_preprocessor=None, 
        num_jobs=1
    ):
        super().__init__(
            X, y, loss_metric, n_splits=n_splits, n_repeats=n_repeats, 
            holdout=holdout, task=task, scale_output=scale_output, 
            input_preprocessor=input_preprocessor, num_jobs=num_jobs
        )

        self.estimator = estimator
        self.needs_proba = needs_proba
    
    def construct_model(self,params):
        return clone(self.estimator).set_params(**params)
    
    def _fit_and_test(self, params, train_index, test_index):
        model = self.construct_model(params)

        X_train = self.X[train_index,...];X_test = self.X[test_index,...]

        if self.input_preprocessor is not None:
            input_preprocessor = clone(self.input_preprocessor).fit(self.X[train_index,...])
            X_train = input_preprocessor.transform(X_train)
            X_test = input_preprocessor.transform(X_test)

        scorer = make_scorer(self.loss_metric,needs_proba=self.needs_proba)

        # fit model
        model.fit(X_train,self.y[train_index,...])

        return scorer(model,X_test,self.y[test_index,...])