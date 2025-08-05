import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from joblib import Parallel, delayed
from typing import Callable, List, Optional, Dict, Union

class CVObjective:
    """Base class for cross-validation objective functions.

    This class provides repeated K-fold or stratified repeated K-fold cross-validation
    for regression and classification tasks, with options for holdout evaluation,
    output scaling, input preprocessing, and parallel execution.

    Subclasses must override:
      - construct_model(): build a model given hyperparameters.
      - _fit_and_test(): train and evaluate the model on a single split.

    Args:
        X: Feature data of shape (n_samples, n_features).
        y: Target data of shape (n_samples,). Classification labels are encoded internally.
        task: One of 'regression', 'binary_classification', or 'classification'.
        loss_metric: Function that computes loss given (y_true, y_pred).
        n_splits: Number of folds per cross-validation repeat. Defaults to 5.
        n_repeats: Number of times to repeat the cross-validation. Defaults to 5.
        holdout: If True, evaluate only the first fold. Defaults to False.
        scale_output: If True and task='regression', standardize target values in training data.
            Defaults to False.
        input_preprocessor: Preprocessing transformer fit and applied per split. Defaults to None.
        stratified: If True and task is either 'binary_classification', or 'classification', use 
            stratified K-fold splits. Defaults to True.
        num_jobs: Number of parallel jobs for fold evaluations. Defaults to 1.
    """
    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        task: str,
        loss_metric: Callable,
        n_splits: int = 5,
        n_repeats: int = 5,
        holdout: bool = False,
        scale_output: bool = False,
        input_preprocessor=None,
        stratified: bool = True,
        num_jobs: int = 1
    ):
        self.X = X
        self.y = y
        self.task = task
        self.loss_metric = loss_metric
        self.stratified = stratified

        # select appropriate splitter
        if self.stratified and 'classification' in self.task:
            self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
        else:
            self.cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

        # generate splits, optionally holdout
        all_splits = list(self.cv.split(self.X, self.y))
        self.train_test_splits = all_splits[:1] if holdout else all_splits
        self.holdout = holdout

        # encode classification labels
        if 'classification' in self.task:
            self.y = LabelEncoder().fit_transform(self.y)

        self.scale_output = scale_output
        self.input_preprocessor = input_preprocessor
        self.num_jobs = num_jobs

    def construct_model(self, params: Dict, **kwargs):
        """Construct and return a model given hyperparameters.

        Must be implemented by subclasses.

        Args:
            params: Hyperparameter name to value mapping.
            **kwargs: Additional arguments for model construction.
        Returns:
            object: An untrained model instance.
        """
        raise NotImplementedError("Subclasses must implement construct_model.")

    def _fit_and_test(self, params: Dict, train_index: List[int], test_index: List[int]) -> float:
        """Train the model on one split and return its loss.

        Must be implemented by subclasses.

        Args:
            params: Hyperparameter mapping for the model.
            train_index: Indices for the training samples.
            test_index: Indices for the testing samples.
        Returns:
            float: Loss computed by loss_metric on test data.
        """
        raise NotImplementedError("Subclasses must implement _fit_and_test.")

    def __call__(
        self,
        params: Dict,
        fold_idxs: Optional[List[int]] = None,
        all: bool = False
    ):
        """Compute cross-validation loss for given hyperparameters.

        Args:
            params: Dictionary of hyperparameters.
            fold_idxs: Indices of folds to evaluate.
                Defaults to all folds or first if holdout.
            all: If True, return array of losses per fold;
                otherwise return mean loss. Defaults to False.
        Returns:
            float or np.ndarray: Mean loss (if all=False) or array of per-fold losses.
        """
        # determine folds
        n_folds = self.cv.get_n_splits(self.X, self.y)
        if fold_idxs is None:
            fold_idxs = [0] if self.holdout else list(range(n_folds))

        # compute losses in parallel
        losses = Parallel(n_jobs=self.num_jobs)(
            delayed(self._fit_and_test)(
                params,
                self.train_test_splits[i][0],
                self.train_test_splits[i][1]
            ) for i in fold_idxs
        )

        losses_arr = np.array(losses)
        return losses_arr if all else float(losses_arr.mean())

    def cvloss(self,
        params: Dict,
        fold_idxs: Optional[List[int]] = None,
        all: bool = False
    ) -> Union[float, np.ndarray]:
        '''
        Compute cross-validation loss for given hyperparameters (deprecated alias).
        
        .. deprecated:: 0.3.0
           Use the callable interface of this object instead, e.g.
           ``losses = obj(params, fold_idxs=fold_idxs, all=all)``.

        Args:
            params: Dictionary of hyperparameters.
            fold_idxs: Indices of folds to evaluate.
                Defaults to all folds or first if holdout.
            all: If True, return array of losses per fold;
                otherwise return mean loss. Defaults to False.
        Returns:
            float or np.ndarray: Mean loss (if all=False) or array of per-fold losses.
        '''
        return self(params, fold_idxs=fold_idxs, all=all)