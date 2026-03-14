import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from typing import Dict, List

from fcvopt.crossvalidation import SklearnCVObj


class XGBoostCVObjEarlyStopping(SklearnCVObj):
    """Cross-validation objective with per-fold early stopping for XGBoost
    (scikit-learn API).

    Extends :class:`SklearnCVObj` by creating an internal validation split within
    each training fold and supplying it to XGBoost via ``eval_set`` together with
    ``early_stopping_rounds``. The outer test fold remains untouched, providing a
    clean generalization estimate.

    Args:
        early_stopping_rounds: Number of rounds without improvement on the inner
            validation split before stopping.
        validation_split: Fraction of each training fold held out for early stopping.
        **kwargs: Forwarded to :class:`SklearnCVObj`.
    """
    def __init__(
        self,
        early_stopping_rounds: int,
        validation_split: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_split = validation_split

    def fit_and_test(
        self,
        params: Dict,
        train_index: List[int],
        test_index: List[int]
    ) -> float:
        # prepare model
        model = self.construct_model(params)
        model.set_params(early_stopping_rounds=self.early_stopping_rounds)

        # slice data
        if isinstance(self.X, pd.DataFrame):
            X_train = self.X.iloc[train_index]
            X_test = self.X.iloc[test_index]
        else:
            X_train = self.X[train_index]
            X_test = self.X[test_index]

        y_train = self.y[train_index]
        y_test = self.y[test_index]

        # optional output scaling
        if self.scale_output and 'regression' in self.task:
            mean, std = y_train.mean(), y_train.std()
            y_train = (y_train - mean) / std

        # optional input preprocessing
        if self.input_preprocessor is not None:
            prep = clone(self.input_preprocessor).fit(X_train, y_train)
            X_train = prep.transform(X_train)
            X_test = prep.transform(X_test)

        # split for early stopping
        stratify = y_train if self.stratified and 'classification' in self.task else None
        X_train2, X_val, y_train2, y_val = train_test_split(
            X_train, y_train,
            test_size=self.validation_split,
            stratify=stratify,
            random_state=None
        )

        # fit with early stopping
        model.fit(
            X_train2, y_train2,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # score on test set
        scorer = make_scorer(self.loss_metric, needs_proba=self.needs_proba)
        return scorer(model, X_test, y_test)
