import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
from typing import Callable, List, Optional, Dict, Union

from ..crossvalidation.cvobjective import CVObjective

class SklearnCVObj(CVObjective):
    """Cross‑validation objective for general scikit‑learn estimators.

    Wraps an unfitted scikit‑learn estimator and evaluates it using the fold infrastructure 
    provided by :class:`CVObjective`. The estimator must implement ``fit`` and ``predict``; 
    if you set ``needs_proba=True`` and your loss uses probabilities, it should also implement 
    ``predict_proba``.

    See :class:`CVObjective` for fold selection, aggregation behavior, and leakage
    safeguards.
    
    Args:
        estimator: Unfitted model object conforming to scikit-learn estimator API. The estimator
            should implement `fit` and `predict`, and optionally `predict_proba` for classification tasks
            if the `needs_proba` flag is set to True and the loss metric requires probabilities.
        X: Feature data of shape (n_samples, n_features).
        y: Target data of shape (n_samples,).
        task: One of 'regression' or 'classification'.
        loss_metric: Function that computes a loss given (y_true, y_pred).
        needs_proba: Whether the metric requires class probabilities. If True, the estimator's `predict_proba`
            method is used instead of `predict`. Applicable only for classification tasks. Defaults to False.
        n_splits: Number of folds for cross-validation. Defaults to 10.
        n_repeats: Number of CV repeats. Defaults to 1.
        scale_output: If True and task='regression', target values are standardized
            per training fold. Defaults to False.
        input_preprocessor: Optional scikit-learn input transformer fit and applied per split. 
            Defaults to None.
        stratified: If True and task is 'classification', use stratified K-fold splits. Defaults to True.
        num_jobs: Number of parallel jobs for fold evaluations. Defaults to 1.
        rng_seed: Random seed for the `estimator` random state, if applicable.
    
    Example:
        .. code-block:: python
        
            from sklearn.ensemble import RandomForestClassifer
            from sklearn.metrics import accuracy_score
            from fcvopt.crossvalidation import SklearnCVObj

            # loss metric: misclassification rate
            def misclass_rate(y_true, y_pred):
                return 1 - accuracy_score(y_true, y_pred)
            

            X, y = load_breast_cancer(return_X_y=True)
            estimator = RandomForestClassifier()

            # Create the cross-validation objective
            # 1 repeat, 10 folds
            cv_obj = SklearnCVObj(
                estimator, X, y,
                task='classification',
                loss_metric=misclass_rate,
                n_splits=10,
                rng_seed=42
            )

            # 10-fold cv loss for a set of hyperparameters
            params = {'n_estimators': 100, 'max_depth': 5}
            mcr = cv_obj(params)
            print(f'Misclassification rate for hyperparameters {params}: {mcr:.4f}')

            # per-fold misclassification rates
            fold_losses = cv_obj(params, all=True)
            print(f'Per-fold misclassification rates for hyperparameters {params}: {fold_losses}')
    """
    def __init__(
        self,
        estimator: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        task: str,
        loss_metric: Callable,
        needs_proba: bool = False,
        n_splits: int = 10,
        n_repeats: int = 1,
        scale_output:bool = False,
        input_preprocessor:TransformerMixin=None,
        stratified: bool = False,
        num_jobs: int = 1,
        rng_seed: Optional[int] = None
    ):
        # Initialize with estimator and cross-validation settings
        super().__init__(
            X=X,
            y=y,
            task=task,
            loss_metric=loss_metric,
            n_splits=n_splits,
            n_repeats=n_repeats,
            stratified=stratified,
            num_jobs=num_jobs,
            rng_seed=rng_seed
        )
        self.estimator = estimator
        self.needs_proba = needs_proba
        self.scale_output = scale_output
        self.input_preprocessor = input_preprocessor
        self._rng = np.random.default_rng(rng_seed)

    def construct_model(self, params: Dict) -> BaseEstimator:
        """
        Clone the base estimator, set provided hyperparameters, and (if supported)
        assign a deterministic ``random_state`` derived from ``rng_seed``.

        Cloning ensures no state leaks across folds. A distinct, reproducible seed
        is generated for each fit when the estimator exposes a ``random_state`` parameter.

        Args:
            params: Hyperparameter name → value mapping.

        Returns:
            A fresh, unfitted estimator configured with ``params`` (and possibly
            ``random_state``).
        """
        model = clone(self.estimator).set_params(**params)
        # assign reproducible seed if supported
        if hasattr(model, 'random_state'):
            seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
            model.set_params(random_state=seed)
        return model

    def fit_and_test(
        self,
        params: Dict,
        train_index: List[int],
        test_index: List[int]
    ) -> float:
        """Fit on the training split and return the loss on the test split.

        Steps performed:
        
        1) Slice ``X``/``y`` by the provided indices.
        2) If ``input_preprocessor`` is set, clone + fit on **train only**, then transform
           both train and test.
        3) If ``scale_output`` and regression, standardize targets using **train**
           statistics.
        4) Build a scorer via :func:`sklearn.metrics.make_scorer`, using probabilities
           when ``needs_proba=True``.
        5) Fit the estimator and compute loss on the test slice.

        Args:
            params: Hyperparameters forwarded to :meth:`construct_model`.
            train_index: Row indices for the training portion of this split.
            test_index: Row indices for the testing portion of this split.

        Returns:
            Scalar loss for this split (lower is better).
        """
        model = self.construct_model(params)

        # slice features
        if isinstance(self.X, pd.DataFrame):
            X_train = self.X.iloc[train_index]
            X_test = self.X.iloc[test_index]
        else:
            X_train = self.X[train_index]
            X_test = self.X[test_index]

        # preprocess inputs if requested
        if self.input_preprocessor is not None:
            prep = clone(self.input_preprocessor).fit(X_train, self.y[train_index])
            X_train = prep.transform(X_train)
            X_test = prep.transform(X_test)

        # slice targets
        y_train = self.y[train_index]
        y_test = self.y[test_index]

        # scale regression outputs if requested
        if self.scale_output and 'regression' in self.task:
            mean, std = y_train.mean(), y_train.std()
            y_train = (y_train - mean) / std
            # note: y_test scaling does not affect loss if metric uses raw scale

        # define scorer (scikit-learn >= 1.4: response_method replaces needs_proba)
        if self.needs_proba:
            scorer = make_scorer(self.loss_metric, response_method='predict_proba')
        else:
            scorer = make_scorer(self.loss_metric)

        # fit and score
        model.fit(X_train, y_train)
        return scorer(model, X_test, y_test)
