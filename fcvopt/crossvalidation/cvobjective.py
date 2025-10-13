import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from joblib import Parallel, delayed
from typing import Callable, List, Optional, Dict, Union

class CVObjective:
    """
    Base class for cross-validation objective functions.

    This class provides a framework for evaluating models via K-fold or stratified K-fold
    cross-validation, with support for:

    * Regression and classification tasks (classification labels encoded internally)
    * Optional holdout evaluation (evaluate only the first fold)
    * Optional per-fold output scaling (for regression)
    * Optional per-fold input preprocessing (fitted only on training split)
    * Parallel execution of folds

    **Intended usage:**  
    Subclasses must override:

        - :meth:`construct_model`: Build and return a model given hyperparameters.
        - :meth:`fit_and_test`: Fit the model on one train/test split and return a loss.

    See :class:`SklearnCVObj` for an example.

    The callable interface (:meth:`__call__`) runs cross-validation for a given set of
    hyperparameters and returns either:

        * A NumPy array of per-fold losses (if ``all=True``), or
        * An aggregate (mean) loss over the selected folds (if ``all=False``).

    The folds used for evaluation can be:

        * All folds from the generated CV splits (default)
        * Only the first fold (if ``holdout=True``)
        * An explicit subset via the ``fold_idxs`` argument to :meth:`__call__`

    Args:
        X: Feature data of shape ``(n_samples, n_features)``.
        y: Target data of shape ``(n_samples,)`` or compatible. For classification,
            labels are encoded internally with :class:`sklearn.preprocessing.LabelEncoder`.
        task: One of ``'regression'``, ``'binary_classification'``, or ``'classification'``.
        loss_metric: Callable that computes a loss given ``(y_true, y_pred)``.
        n_splits: Number of folds per CV repeat. Defaults to ``5``.
        n_repeats: Number of CV repeats. Defaults to ``5``.
        holdout: If ``True``, evaluate only the first fold. Defaults to ``False``.
        scale_output: If ``True`` and ``task='regression'``, standardize target values
            **per training fold** before fitting. Defaults to ``False``.
        input_preprocessor: Optional scikit-learn-style transformer to fit and apply
            **within each fold** (avoiding data leakage). Defaults to ``None``.
        stratified: If ``True`` and ``task`` is a classification type, use stratified
            CV splits. Defaults to ``True``.
        num_jobs: Number of parallel jobs for fold evaluations. Defaults to ``1``.

    Notes:
        - The meaning of "loss" is determined entirely by the ``loss_metric`` you provide.
          If you want to optimize a score where higher is better, wrap it into a loss
          (e.g., ``lambda y, yhat: -roc_auc_score(y, yhat)``).
        - Subclasses may choose to aggregate per-fold results differently or compute
          additional statistics.
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
        """
        Build and return an unfitted model for a given hyperparameter configuration.

        Must be implemented by subclasses.

        Args:
            params: Mapping from hyperparameter name to value.
            **kwargs: Optional extras a subclass may accept for construction.

        Returns:
            An unfitted model instance compatible with this objective.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement construct_model.")

    def fit_and_test(self, params: Dict, train_index: List[int], test_index: List[int]) -> float:
        """
        Fit/evaluate on a single CV split and return the loss for that split.

        Must be implemented by subclasses. A typical implementation should:
        
        1) Slice ``X``/``y`` by ``train_index`` and ``test_index``.
        2) Fit and apply ``input_preprocessor`` **on the training slice only**, then
           transform the test slice (to avoid leakage).
        3) If ``scale_output`` and regression task: standardize targets using **training**
           statistics, then fit.
        4) Train the model built by :meth:`construct_model`.
        5) Compute and return the scalar loss via ``loss_metric``.

        Args:
            params: Hyperparameter mapping for model construction.
            train_index: Row indices for the training portion of this split.
            test_index: Row indices for the testing portion of this split.

        Returns:
            Scalar loss for this split (lower is better).

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement fit_and_test.")

    def __call__(
        self,
        params: Dict,
        fold_idxs: Optional[List[int]] = None,
        all: bool = False
    ) -> Union[float, np.ndarray]:
        """Evaluate a hyperparameter configuration on selected CV folds.

        By default, uses all generated folds, unless this objective was created with
        ``holdout=True`` (then only the first fold is used). You can override the default
        by providing ``fold_idxs`` (indices into the internally stored fold list).

        Computation is parallelized across folds according to ``num_jobs``.

        Args:
            params: Hyperparameters to pass to :meth:`construct_model`.
            fold_idxs: Optional list of fold indices to evaluate (e.g., ``[0, 3, 4]``).
                If omitted, uses all folds, or only the first if ``holdout=True``.
            all: If ``True``, return the per‑fold loss array; if ``False``, return
                the aggregate (mean) loss over the selected folds.

        Returns:
            float or ndarray: Mean loss across the selected folds (if ``all=False``),
            otherwise an array of per‑fold losses.

        Notes:
            - The definition of "loss" is entirely determined by ``loss_metric``.
              If you want to optimize a score where higher is better, wrap it into a loss
              (e.g., negative score or ``1 - score``).
            - ``fold_idxs`` refer to the order produced by the internal splitter
              (``RepeatedKFold`` or ``RepeatedStratifiedKFold``).
        """
        # determine folds
        n_folds = self.cv.get_n_splits(self.X, self.y)
        if fold_idxs is None:
            fold_idxs = [0] if self.holdout else list(range(n_folds))

        # compute losses in parallel
        losses = Parallel(n_jobs=self.num_jobs)(
            delayed(self.fit_and_test)(
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