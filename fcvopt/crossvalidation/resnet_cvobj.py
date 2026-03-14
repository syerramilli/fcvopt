import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from typing import Optional, Dict

from ..crossvalidation.cvobjective import CVObjective
from ..configspace import ConfigurationSpace
from ConfigSpace import Float, Integer, Categorical


def _make_normalization(normalization: str, input_dim: int) -> nn.Module:
    """
    Return a normalization layer instance for 1D tabular features.

    Args:
        normalization: Either ``'batchnorm'`` or ``'layernorm'``.
        input_dim: Size of the last (feature) dimension to normalize.

    Returns:
        An initialized :class:`torch.nn.BatchNorm1d` or :class:`torch.nn.LayerNorm`.

    Raises:
        ValueError: If ``normalization`` is not supported.
    """
    mapping = {
        "batchnorm": nn.BatchNorm1d,
        "layernorm": nn.LayerNorm,
    }
    try:
        return mapping[normalization](input_dim)
    except KeyError as exc:
        raise ValueError("normalization must be 'batchnorm' or 'layernorm'") from exc


class ResNetBlock(nn.Module):
    """
    Residual block for a feed-forward network with dropout (tabular data).

    The block computes::

        x + Dropout( Linear( Dropout( ReLU( Linear( Norm(x) ) ) ) ) )

    where ``Norm`` is either batch normalization or layer normalization.

    See `Gorishniy et al. (2021) <https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf>`_
    for details.

    Args:
        input_dim: Last dimension of the input tensor.
        normalization: ``'batchnorm'`` or ``'layernorm'``.
        hidden_factor: Hidden width inside the block is ``floor(hidden_factor * input_dim)``.
        hidden_dropout: Dropout rate inside the hidden path.
        residual_dropout: Dropout rate applied to the residual output.
    """
    def __init__(
        self,
        input_dim: int,
        normalization: str,
        hidden_factor: float = 2.0,
        hidden_dropout: float = 0.1,
        residual_dropout: float = 0.05,
    ):
        super().__init__()
        d_hidden = int(hidden_factor * input_dim)

        self.ff = nn.Sequential(
            _make_normalization(normalization, input_dim),
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(d_hidden, input_dim),
            nn.Dropout(residual_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(x)


class TabularResNet(nn.Module):
    """
    Tabular ResNet model.

    A shallow fully connected stem followed by several residual blocks and a
    prediction head (Norm → ReLU → Linear).

    .. note::
        This implementation expects **all features to be numeric**. Preprocess
        categorical columns (e.g., one-hot or target encoding) beforehand.

    See `Gorishniy et al. (2021) <https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf>`_
    for more details.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension (``1`` for regression/binary classification,
            or number of classes for multiclass).
        n_hidden: Number of residual blocks (default: ``2``).
        layer_size: Width of the hidden representation (default: ``64``).
        normalization: ``'batchnorm'`` or ``'layernorm'``.
        hidden_factor: Expansion factor inside each residual block
            (hidden width ``= floor(hidden_factor * layer_size)``).
        hidden_dropout: Dropout rate inside residual blocks.
        residual_dropout: Dropout rate on the residual output.

    Shape:
        - Input: ``(N, input_dim)``
        - Output: ``(N, output_dim)``

    Attributes:
        ff: Input stem (``Linear(input_dim, layer_size)``) followed by residual blocks.
        prediction: Norm → ReLU → Linear head to ``output_dim``.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden: int = 2,
        layer_size: int = 64,
        normalization: str = "batchnorm",
        hidden_factor: float = 2.0,
        hidden_dropout: float = 0.1,
        residual_dropout: float = 0.05,
    ):
        super().__init__()

        if normalization not in {"batchnorm", "layernorm"}:
            raise ValueError("normalization must be 'batchnorm' or 'layernorm'")

        self.ff = nn.Sequential(nn.Linear(input_dim, layer_size))
        for _ in range(n_hidden):
            self.ff.append(
                ResNetBlock(
                    input_dim=layer_size,
                    normalization=normalization,
                    hidden_factor=hidden_factor,
                    hidden_dropout=hidden_dropout,
                    residual_dropout=residual_dropout,
                )
            )

        self.prediction = nn.Sequential(
            _make_normalization(normalization, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction(self.ff(x))


class ResNetCVObj(CVObjective):
    """
    Cross-validation objective for tabular ResNet models
    (`Gorishniy et al. (2021) <https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf>`_).

    Implements :meth:`fit_and_test` with a self-contained PyTorch training loop
    that includes early stopping, gradient clipping, and learning-rate scheduling.
    No external training library is required.

    Training details per fold:

    * A 10 % internal validation split is held out from the training data to monitor
      the validation loss.
    * Early stopping (configurable ``patience``, default 10) restores the best checkpoint.
    * ``ReduceLROnPlateau`` (factor 0.1, patience 5, min_lr 1e-5) adjusts the
      learning rate during training.
    * Gradient norms are clipped at 5.0 to improve stability.
    * Singleton mini-batches are dropped to avoid ``BatchNorm1d`` failures.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``. All features
            must be numeric; encode categoricals beforehand.
        y: Target array of shape ``(n_samples,)``.
        task: One of ``'regression'``, ``'binary_classification'``, or
            ``'classification'``.
        loss_metric: Callable ``(y_true, y_pred) -> float`` (lower is better).
        needs_proba: If ``True``, probabilities (sigmoid for binary, softmax for
            multiclass) are passed to ``loss_metric`` instead of hard labels.
            Ignored for regression. Defaults to ``False``.
        n_splits: Number of CV folds. Defaults to ``10``.
        n_repeats: Number of CV repeats. Defaults to ``1``.
        stratified: Use stratified splits for classification. Defaults to ``True``.
        scale_output: Standardize regression targets per fold (using training
            statistics only). Defaults to ``False``.
        input_preprocessor: Optional sklearn-compatible transformer fitted on
            each training fold and applied to both train and test. Defaults to ``None``.
        num_jobs: Parallel fold evaluations. Defaults to ``1``.
        rng_seed: Seed for reproducibility. Defaults to ``None``.
        max_epochs: Maximum training epochs per fold. Defaults to ``100``.
        patience: Early-stopping patience (number of epochs without val-loss
            improvement before training halts). Defaults to ``10``.
        optimizer: Name of a ``torch.optim`` optimizer class (e.g. ``'AdamW'``,
            ``'Adam'``, ``'SGD'``). Defaults to ``'AdamW'``.
        batch_size: Mini-batch size. Pass this argument (not a hyperparameter) if
            you want a fixed batch size; include it in the config space only if you
            want to tune it. Defaults to ``256``.
        device: PyTorch device string, e.g. ``'cpu'`` or ``'cuda'``.
            Defaults to ``'cpu'``.

    Expected keys in ``params`` dict (passed to :meth:`fit_and_test` via the optimizer):

    .. list-table::
       :header-rows: 1

       * - Key
         - Description
       * - ``n_hidden``
         - Number of residual blocks
       * - ``layer_size``
         - Hidden width
       * - ``normalization``
         - ``'batchnorm'`` or ``'layernorm'``
       * - ``hidden_factor``
         - Expansion factor inside each block
       * - ``hidden_dropout``
         - Dropout rate inside blocks
       * - ``residual_dropout``
         - Dropout rate on residual output
       * - ``lr``
         - Learning rate
       * - ``weight_decay``
         - L2 regularization strength
       * - ``momentum``
         - Momentum (only when ``optimizer='SGD'``)

    Example:
        .. code-block:: python

            from sklearn.datasets import make_classification
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import StandardScaler
            from fcvopt.crossvalidation import ResNetCVObj
            from fcvopt.optimizers import FCVOpt

            X, y = make_classification(n_samples=1000, n_features=20, random_state=0)

            cv_obj = ResNetCVObj(
                X=X, y=y,
                task='binary_classification',
                loss_metric=lambda yt, yp: 1 - roc_auc_score(yt, yp),
                needs_proba=True,
                n_splits=5,
                input_preprocessor=StandardScaler(),
                max_epochs=100,
            )

            config = cv_obj.get_recommended_configspace()
            optimizer = FCVOpt(obj=cv_obj, n_folds=5, config=config, acq_function='LCB')
            best = optimizer.optimize(n_trials=30)
    """
    def __init__(
        self,
        X,
        y,
        task: str,
        loss_metric,
        needs_proba: bool = False,
        n_splits: int = 10,
        n_repeats: int = 1,
        stratified: bool = True,
        scale_output: bool = False,
        input_preprocessor=None,
        num_jobs: int = 1,
        rng_seed: Optional[int] = None,
        max_epochs: int = 100,
        patience: int = 10,
        optimizer: str = "AdamW",
        batch_size: int = 256,
        device: str = "cpu",
    ):
        super().__init__(
            X=X, y=y, task=task, loss_metric=loss_metric,
            n_splits=n_splits, n_repeats=n_repeats,
            stratified=stratified, num_jobs=num_jobs, rng_seed=rng_seed,
        )

        self.needs_proba = needs_proba
        self.scale_output = scale_output
        self.input_preprocessor = input_preprocessor
        self.max_epochs = max_epochs
        self.patience = patience
        self.optimizer_name = optimizer
        self.batch_size = batch_size
        self.device = torch.device(device)
        self._rng = np.random.default_rng(rng_seed)

        # Determine output dimension and cast targets
        self.num_targets = 1
        if self.task == "classification":
            # y is already int64 after LabelEncoder in CVObjective.__init__
            self.num_targets = int(np.unique(self.y).size)
        elif self.task == "binary_classification":
            # y is int64 {0,1} after LabelEncoder; BCEWithLogitsLoss needs float32
            self.y = self.y.astype(np.float32)

        self.input_dim = int(np.asarray(self.X).shape[1])

    def construct_model(self, params: Dict) -> TabularResNet:
        """
        Build and return an uninitialized :class:`TabularResNet` from ``params``.

        Args:
            params: Hyperparameter mapping; must contain ``n_hidden``,
                ``layer_size``, ``normalization``, ``hidden_factor``,
                ``hidden_dropout``, ``residual_dropout``.

        Returns:
            An initialized :class:`TabularResNet` placed on ``self.device``.
        """
        return TabularResNet(
            input_dim=self.input_dim,
            output_dim=self.num_targets,
            n_hidden=params["n_hidden"],
            layer_size=params["layer_size"],
            normalization=params["normalization"],
            hidden_factor=params["hidden_factor"],
            hidden_dropout=params["hidden_dropout"],
            residual_dropout=params["residual_dropout"],
        ).to(self.device)

    def evaluate(self, model: "TabularResNet", X: np.ndarray) -> torch.Tensor:
        """
        Run a trained model on a feature array and return predictions as a CPU tensor.

        Task-specific output transformations are applied so the result is ready
        to pass directly to ``loss_metric``:

        * **Regression**: raw output, shape ``(N,)``
        * **Binary classification**: sigmoid of the logit, shape ``(N,)``
        * **Multiclass classification**: class probabilities via softmax, shape ``(N, n_classes)``

        Args:
            model: A trained :class:`TabularResNet` instance.
            X: Feature array of shape ``(N, n_features)``; will be cast to float32.

        Returns:
            Prediction tensor on CPU.
        """
        dataset = TensorDataset(torch.from_numpy(np.asarray(X).astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        model.eval()
        chunks = []
        with torch.no_grad():
            for (X_batch,) in loader:
                chunks.append(model(X_batch.to(self.device)).cpu())
        out = torch.cat(chunks, dim=0)

        if self.task == "regression":
            return out.squeeze(-1)
        if self.task == "binary_classification":
            return torch.sigmoid(out).squeeze(-1)
        return torch.softmax(out, dim=1)

    def fit_and_test(
        self,
        params: Dict,
        train_index,
        test_index,
    ) -> float:
        """
        Train a :class:`TabularResNet` on one CV fold and return the test loss.

        Steps:

        1. Slice ``X``/``y`` by ``train_index`` / ``test_index``.
        2. Apply ``input_preprocessor`` (fit on train only) if provided.
        3. Standardize regression targets using train statistics if
           ``scale_output=True``.
        4. Hold out 10 % of the training data as an internal validation set.
        5. Train via a :class:`~torch.utils.data.DataLoader` (early stopping +
           gradient clipping + ``ReduceLROnPlateau``).
        6. Restore the best checkpoint; compute the test metric via
           :meth:`evaluate`.

        Args:
            params: Hyperparameter configuration.
            train_index: Row indices for the training portion of this split.
            test_index: Row indices for the testing portion of this split.

        Returns:
            Scalar test loss for this fold (lower is better).
        """
        X = np.asarray(self.X)

        X_train_full = X[train_index].astype(np.float32)
        X_test = X[test_index].astype(np.float32)
        y_train_full = self.y[train_index].copy()
        y_test = self.y[test_index]

        # Optional input preprocessing (fit on train only)
        if self.input_preprocessor is not None:
            prep = clone(self.input_preprocessor).fit(X_train_full)
            X_train_full = prep.transform(X_train_full).astype(np.float32)
            X_test = prep.transform(X_test).astype(np.float32)

        # Optional output scaling for regression
        y_mean, y_std = 0.0, 1.0
        if self.scale_output and self.task == "regression":
            y_arr = y_train_full.astype(np.float64)
            y_mean = float(y_arr.mean())
            y_std = float(y_arr.std()) or 1.0
            y_train_full = ((y_arr - y_mean) / y_std).astype(np.float32)

        # Internal validation split (10 %)
        stratify = y_train_full.astype(np.int64) if "classification" in self.task else None
        val_seed = int(self._rng.integers(0, 2**31))
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=0.1, stratify=stratify, random_state=val_seed,
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=0.1, stratify=None, random_state=val_seed,
            )

        # Build model, loss, optimizer, scheduler
        model = self.construct_model(params)

        if self.task == "regression":
            criterion = nn.MSELoss()
            val_criterion = nn.MSELoss(reduction="sum")
        elif self.task == "binary_classification":
            criterion = nn.BCEWithLogitsLoss()
            val_criterion = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            criterion = nn.CrossEntropyLoss()
            val_criterion = nn.CrossEntropyLoss(reduction="sum")

        opt_cls = getattr(torch.optim, self.optimizer_name)
        opt_kwargs: Dict = {"lr": params["lr"], "weight_decay": params["weight_decay"]}
        if self.optimizer_name == "SGD" and "momentum" in params:
            opt_kwargs["momentum"] = params["momentum"]
        opt = opt_cls(model.parameters(), **opt_kwargs)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.1, patience=5, min_lr=1e-5,
        )

        # Build DataLoaders — data stays on CPU; batches are moved to device on the fly
        dev = self.device

        if self.task in ("regression", "binary_classification"):
            y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).reshape(-1, 1)
            y_val_t = torch.from_numpy(y_val.astype(np.float32)).reshape(-1, 1)
        else:
            y_tr_t = torch.from_numpy(y_tr.astype(np.int64))
            y_val_t = torch.from_numpy(y_val.astype(np.int64))

        train_dataset = TensorDataset(torch.from_numpy(X_tr), y_tr_t)
        val_dataset = TensorDataset(torch.from_numpy(X_val), y_val_t)

        # drop_last avoids singleton batches that break BatchNorm1d
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > self.batch_size,
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        patience_counter = 0

        for _ in range(self.max_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(dev), y_batch.to(dev)
                opt.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for X_vb, y_vb in val_loader:
                    X_vb, y_vb = X_vb.to(dev), y_vb.to(dev)
                    val_loss_sum += val_criterion(model(X_vb), y_vb).item()
            val_loss = val_loss_sum / len(val_dataset)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

        # Predict and compute metric via evaluate()
        y_pred = self.evaluate(model, X_test).numpy()

        if self.task == "regression":
            if self.scale_output:
                y_pred = y_pred * y_std + y_mean
            return self.loss_metric(y_test, y_pred)

        if self.task == "binary_classification":
            y_true = y_test.astype(np.int64)
            if self.needs_proba:
                return self.loss_metric(y_true, y_pred)
            return self.loss_metric(y_true, (y_pred > 0.5).astype(np.int64))

        # multiclass classification
        if self.needs_proba:
            return self.loss_metric(y_test, y_pred)
        return self.loss_metric(y_test, y_pred.argmax(axis=1))

    def get_recommended_configspace(self) -> "ConfigurationSpace":
        """
        Recommended hyperparameter search space for Tabular ResNet.

        Hyperparameters:
            - n_hidden: Integer, log-uniform in [1, 6]
            - layer_size: Integer, log-uniform in [8, 512]
            - normalization: Categorical in {'batchnorm', 'layernorm'}
            - hidden_factor: Float in [1.0, 4.0]
            - hidden_dropout: Float in [0.0, 0.5]
            - residual_dropout: Float in [0.0, 0.5]
            - lr: Float, log-uniform in [1e-5, 1e-1]
            - weight_decay: Float, log-uniform in [1e-8, 1e-2]

        Returns:
            ConfigurationSpace: A config space ready to plug into your optimizer.
        """
        config = ConfigurationSpace()

        # Architecture
        config.add(Integer("n_hidden", lower=1, upper=6, log=True, default=2))
        config.add(Integer("layer_size", lower=8, upper=512, log=True, default=64))
        config.add(Categorical("normalization", choices=["batchnorm", "layernorm"], default="batchnorm"))
        config.add(Float("hidden_factor", lower=1.0, upper=4.0, default=2.0))
        config.add(Float("hidden_dropout", lower=0.0, upper=0.5, default=0.1))
        config.add(Float("residual_dropout", lower=0.0, upper=0.5, default=0.05))

        # Optimization
        config.add(Float("lr", lower=1e-5, upper=1e-1, log=True, default=1e-3))
        config.add(Float("weight_decay", lower=1e-8, upper=1e-2, log=True, default=1e-5))

        return config
