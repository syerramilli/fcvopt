import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from typing import Callable, List, Optional, Dict, Union

from ..crossvalidation.cvobjective import CVObjective

class SklearnCVObj(CVObjective):
    """Cross-validation objective for general scikit-learn estimators.

    The estimator passed must implement `fit` and `predict`, and optionally
    `predict_proba` if probability-based metrics are needed.
    
    Args:
        estimator: Unfitted model object conforming to scikit-learn estimator API. The estimator
            should implement `fit` and `predict`, and optionally `predict_proba` for classification tasks
            if the `needs_proba` flag is set to True and the loss metric requires probabilities.
        X: Feature data of shape (n_samples, n_features).
        y: Target data of shape (n_samples,).
        task: One of 'regression', 'binary_classification', or 'classification'.
        loss_metric: Function that computes a loss given (y_true, y_pred).
        needs_proba: If True, `predict_proba` is used instead of `predict` to obtain
            the prediction probabilities for scoring. Applicable only for classification
            tasks. Defaults to False.
        n_splits: Number of folds for cross-validation. Defaults to 10.
        n_repeats: Number of CV repeats. Defaults to 1.
        holdout: If True, only the first fold is evaluated. Defaults to False.
        scale_output: If True and task='regression', target values are standardized
            per training fold. Defaults to False.
        input_preprocessor: Optional scikit-learn input transformer fit and applied per split. 
            Defaults to None.
        stratified: If True and task is either 'binary_classification', or 'classification', use 
            stratified K-fold splits. Defaults to True.
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
                task='binary_classification',
                loss_metric=misclass_rate,
                rng_seed=42,
            )

            # cv loss for a set of hyperparameters
            params = {'n_estimators': 100, 'max_depth': 5}
            mcr = cv_obj(params)
            print(f'Misclassification rate for hyperparameters {params}: {mcr:.4f}')        
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
        holdout: bool = False,
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
            holdout=holdout,
            scale_output=scale_output,
            input_preprocessor=input_preprocessor,
            stratified=stratified,
            num_jobs=num_jobs
        )
        self.estimator = estimator
        self.needs_proba = needs_proba
        self._rng = np.random.default_rng(rng_seed)

    def construct_model(self, params: Dict) -> BaseEstimator:
        """Clone the scikit-learn estimator with given hyperparameters and set random state.

        Args:
            params: Hyperparameter name to value mapping.

        Returns:
            A new untrained estimator instance with the specified hyperparameters.
        """
        model = clone(self.estimator).set_params(**params)
        # assign reproducible seed if supported
        if hasattr(model, 'random_state'):
            seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
            model.set_params(random_state=seed)
        return model

    def _fit_and_test(
        self,
        params: Dict,
        train_index: List[int],
        test_index: List[int]
    ) -> float:
        """Fit the estimator on a split and return the loss score.

        This method handles data slicing, optional preprocessing,
        optional output scaling, model fitting, and scoring.

        Args:
            params: Hyperparameters for the model.
            train_index: Indices for training samples.
            test_index: Indices for testing samples.

        Returns:
            Loss computed by the configured loss_metric.
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

        # define scorer
        scorer = make_scorer(self.loss_metric, needs_proba=self.needs_proba)

        # fit and score
        model.fit(X_train, y_train)
        return scorer(model, X_test, y_test)

class XGBoostCVObjEarlyStopping(SklearnCVObj):
    """Cross-validation objective with early stopping for XGBoost models (
    scikit-learn API).

    Extends :class:`SklearnCVObj` to include a validation split within each CV fold
    for early stopping based on the loss metric. If early stopping is not required,
    use :class:`SklearnCVObj` directly.

    Args:
        early_stopping_rounds (int):
            Number of rounds without improvement before stopping.
        validation_split (float, optional):
            Fraction of fold training data held out for early stopping. Defaults to 0.1.
        **kwargs: Passed through to the SklearnCVObj initializer, including:
            - estimator: XGBoost regressor or classifier
            - X, y, task, loss_metric, and CV settings
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

    def _fit_and_test(
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