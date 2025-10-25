import numpy as np
import torch
import torch.nn as nn

try:
    from skorch import NeuralNetRegressor, NeuralNetClassifier
    from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
    from skorch.dataset import ValidSplit
except ImportError as e:
    raise ImportError("skorch must be installed to use ResNetCVObj") from e

from sklearn.metrics import make_scorer
from typing import Optional, Dict

from ..crossvalidation.sklearn_cvobj import SklearnCVObj
from ..configspace import ConfigurationSpace
from ConfigSpace import Float, Integer, Categorical


def make_normalization(normalization: str, input_dim: int) -> nn.Module:
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
            make_normalization(normalization, input_dim),
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),  # hidden dropout
            nn.Linear(d_hidden, input_dim),
            nn.Dropout(residual_dropout),  # residual dropout
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
        output_dim: Output dimension (``1`` for regression, or number of classes).
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
            make_normalization(normalization, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prediction(self.ff(x))


class ResNetCVObj(SklearnCVObj):
    """
    Cross-validation objective for tabular ResNet models
    (PyTorch + skorch; `Gorishniy et al. (2021) <https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf>`_).

    Builds a scikit-learn-compatible skorch estimator around :class:`TabularResNet`
    and evaluates it using the CV pipeline from :class:`fcvopt.crossvalidation.SklearnCVObj`.

    Built-ins:
      - **Early stopping** (patience 15) monitored on a validation metric
      - **LR scheduling**: ``ReduceLROnPlateau`` (factor 0.1, patience 5, min_lr 1e-5)
      - Uses :func:`sklearn.metrics.make_scorer` with ``greater_is_better=False`` so
        lower values indicate better loss

    Args:
        max_epochs: Maximum training epochs per fold.
        optimizer: Optimizer name from ``torch.optim`` (e.g., ``'SGD'``, ``'Adam'``, ``'RMSprop'``).
        **kwargs: Forwarded to :class:`SklearnCVObj` (e.g., ``X``, ``y``, ``task``,
            ``loss_metric``, CV settings, ``needs_proba``, etc.).

    Notes:
        * Tasks: ``'regression'``, ``'binary_classification'``, or ``'classification'``.
          For binary classification, the model outputs two logits (softmax recommended at inference).
        * A ``ValidSplit(10, stratified=...)`` is used; stratification is enabled for
          classification tasks.
        * Labels are cast appropriately: int64 for classification.

    Expected keys in ``params`` (for :meth:`construct_model`):
        - ``n_hidden``: number of residual blocks
        - ``layer_size``: hidden width
        - ``normalization``: ``'batchnorm'`` or ``'layernorm'``
        - ``hidden_factor``: expansion factor inside blocks
        - ``hidden_dropout``: dropout inside blocks
        - ``residual_dropout``: dropout on residual output
        - ``lr``: learning rate
        - ``weight_decay``: L2 weight decay
        - ``batch_size``: batch size
        - ``momentum``: (only if ``optimizer == 'SGD'``)
    """
    def __init__(
        self,
        max_epochs: int = 100,
        optimizer: str = "SGD",
        **kwargs,
    ):
        super().__init__(estimator=None, **kwargs)
        self.max_epochs = max_epochs
        self.optimizer = optimizer

        # Determine target formatting / output dimension
        self.num_targets = 1
        if "classification" in self.task:
            self.y = self.y.astype(np.int64)
            if self.task == "classification":
                self.num_targets = int(np.unique(self.y).size)
            elif self.task == "binary_classification":
                self.num_targets = 2  # two logits for CE

        self.input_dim = self.X.shape[1]

    def construct_model(self, params: Dict):
        """
        Build a skorch-wrapped :class:`TabularResNet` configured from ``params`` and class settings.

        Returns:
            A ``NeuralNetRegressor`` or ``NeuralNetClassifier`` ready for ``fit``/``predict``.
        """
        if self.task == "regression":
            skorch_class = NeuralNetRegressor
            criterion = nn.MSELoss
        else:
            # Multiclass or binary classification → CrossEntropyLoss
            skorch_class = NeuralNetClassifier
            criterion = nn.CrossEntropyLoss

        model = skorch_class(
            module=TabularResNet,
            criterion=criterion,
            iterator_train__shuffle=True,
            iterator_train__drop_last=True,
            module__input_dim=self.input_dim,
            module__output_dim=self.num_targets,
            module__n_hidden=params["n_hidden"],
            module__layer_size=params["layer_size"],
            module__normalization=params["normalization"],
            module__hidden_factor=params["hidden_factor"],
            module__hidden_dropout=params["hidden_dropout"],
            module__residual_dropout=params["residual_dropout"],
            callbacks=[
                EpochScoring(
                    scoring=make_scorer(
                        self.loss_metric,
                        needs_proba=self.needs_proba,
                        greater_is_better=False,  # treat metric as loss
                    ),
                    lower_is_better=True,
                    name="valid_metric",
                ),
                EarlyStopping(patience=15, monitor="valid_metric", load_best=True),
                LRScheduler(
                    policy="ReduceLROnPlateau",
                    monitor="valid_metric",
                    factor=0.1,
                    mode="min",
                    patience=5,
                    verbose=False,
                    min_lr=1e-5,
                ),
            ],
            optimizer=getattr(torch.optim, self.optimizer),
            optimizer__lr=params["lr"],
            optimizer__weight_decay=params["weight_decay"],
            max_epochs=self.max_epochs,
            batch_size=params.get("batch_size", 256),
            train_split=ValidSplit(10, stratified=("classification" in self.task)),
            verbose=0,
        )

        if self.optimizer == "SGD" and "momentum" in params:
            model.set_params(optimizer__momentum=params["momentum"])

        return model

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