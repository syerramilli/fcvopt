import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Union

try:
    from skorch import (
        NeuralNetRegressor,
        NeuralNetClassifier,
        NeuralNetBinaryClassifier,
    )
    from skorch.callbacks import (
        GradientNormClipping,
        EarlyStopping,
        EpochScoring,
        LRScheduler,
    )
    from skorch.dataset import ValidSplit
except ImportError as e:
    raise ImportError("skorch must be installed to use MLPCVObj") from e

from sklearn.metrics import make_scorer

from .sklearn_cvobj import SklearnCVObj
from ..configspace import ConfigurationSpace
from ConfigSpace import Float, Integer


class MLP(nn.Module):
    """
    Feed-forward neural network with optional categorical embeddings and dropout.

    Supports:
      - purely numerical inputs
      - mixed numerical + categorical inputs (each categorical feature uses a 2-D embedding)
      - activations: ``'ReLU'``, ``'SELU'`` (with α-dropout), or ``'Sigmoid'``
      - multiple hidden layers with per-layer widths and dropouts

    Args:
        h_sizes: Hidden layer sizes. Length determines the number of layers (>= 1).
        dropouts: Dropout rate per hidden layer. Must have the same length as ``h_sizes``.
        output_dim: Dimension of the final output. Use ``1`` for regression and
            binary classification; for multiclass, set to the number of classes.
        numerical_index: Indices of numerical features in the input vector.
        activation: One of ``{'ReLU', 'SELU', 'Sigmoid'}``. If ``'SELU'``, weights are
            initialized for self-normalizing nets and α-dropout is used. Defaults to
            ``'Sigmoid'``.
        categorical_index: Indices of categorical features to embed (optional).
        num_levels_per_var: Cardinalities for each index in ``categorical_index``.
            Required if ``categorical_index`` is provided.

    Notes:
        * When embeddings are used, each categorical feature is embedded into a
          2-dimensional vector and concatenated with the numerical features.
        * To avoid data leakage, any external preprocessing should be fit on training
          data only (see the cross-validation wrappers for per-fold preprocessing).
    """
    def __init__(
        self,
        h_sizes: List[int],
        dropouts: List[float],
        output_dim: int,
        numerical_index: List[int],
        activation: str = "Sigmoid",
        categorical_index: Optional[List[int]] = None,
        num_levels_per_var: Optional[List[int]] = None,
    ):
        super().__init__()

        # Basic validation
        if len(h_sizes) == 0:
            raise ValueError("h_sizes must contain at least one hidden layer width.")
        if len(h_sizes) != len(dropouts):
            raise ValueError("dropouts must match the length of h_sizes.")
        if activation not in {"ReLU", "SELU", "Sigmoid"}:
            raise ValueError("activation must be one of {'ReLU','SELU','Sigmoid'}.")

        # Embedding setup for categorical variables
        self.embedding_layers = None
        if categorical_index is None:
            input_dim = len(numerical_index)
            self.categorical_index = None
        else:
            if not num_levels_per_var or len(num_levels_per_var) != len(categorical_index):
                raise ValueError(
                    "num_levels_per_var must be provided and match categorical_index length."
                )
            self.categorical_index = torch.tensor(categorical_index, dtype=torch.long)
            self.embedding_layers = nn.ModuleList(
                [nn.Embedding(levels, 2) for levels in num_levels_per_var]
            )
            input_dim = len(numerical_index) + 2 * len(categorical_index)

        self.numerical_index = torch.tensor(numerical_index, dtype=torch.long)

        # Build hidden layers
        layers: List[nn.Module] = []
        for h, d in zip(h_sizes, dropouts):
            layers.append(nn.Linear(input_dim, h))
            layers.append(getattr(nn, activation)())
            layers.append(nn.AlphaDropout(d) if activation == "SELU" else nn.Dropout(d))
            input_dim = h
        self.hidden_layers = nn.Sequential(*layers)

        # Final output layer
        self.output = nn.Linear(h_sizes[-1], output_dim)

        # SELU-friendly initialization (fan_in, gain=1 ≈ LeCun normal)
        if activation == "SELU":
            def init_fn(m: nn.Module):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                    nn.init.zeros_(m.bias)
            self.apply(init_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        If embeddings are configured, the categorical columns are embedded and
        concatenated with the numerical columns before passing through the MLP.
        """
        if self.embedding_layers is not None and self.categorical_index is not None:
            embeds = []
            for idx, emb in enumerate(self.embedding_layers):
                col = self.categorical_index[idx]
                embeds.append(emb(x[..., col].long()))
            x_num = x[..., self.numerical_index]
            x = torch.cat([torch.cat(embeds, dim=1), x_num], dim=1)
        else:
            x = x[..., self.numerical_index]

        x = self.hidden_layers(x)
        return self.output(x)


class MLPCVObj(SklearnCVObj):
    """
    Cross-validation objective for a feed-forward neural network (PyTorch + skorch).

    Wraps :class:`MLP` in a scikit-learn-compatible skorch estimator and evaluates it
    using the CV pipeline from :class:`fcvopt.crossvalidation.SklearnCVObj`.

    Built-ins:
      - **Early stopping**: patience 15 epochs on a validation metric
      - **LR scheduling**: ``ReduceLROnPlateau`` (factor 0.1, patience 5, min_lr 1e-5)
      - **Gradient norm clipping**: cap gradient norm at 5.0

    Args:
        num_hidden: Number of hidden layers. The per-layer hyperparameters are expected
            in ``params`` as ``hsize{i}`` and ``dropout{i}`` for ``i = 0 .. num_hidden-1``.
        activation: Hidden activation (``'ReLU'``, ``'SELU'``, or ``'Sigmoid'``).
        max_epochs: Maximum training epochs per fold.
        optimizer: Name of PyTorch optimizer (``'SGD'``, ``'Adam'``, or ``'RMSprop'``).
        numerical_index: Indices of numerical features. If ``None`` and
            ``categorical_index`` is provided, the numerical indices are inferred as
            the complement of ``categorical_index``. If both are provided, they must
            be disjoint.
        categorical_index: Indices of categorical features to embed (optional).
        num_levels_per_var: Cardinalities for each categorical variable (required if
            ``categorical_index`` is provided).
        **kwargs: Forwarded to :class:`SklearnCVObj` (e.g., ``X``, ``y``, ``task``,
            ``loss_metric``, CV settings, ``needs_proba``, etc.).

    Notes:
        * For multiclass classification (``task='classification'``), targets are cast
          to ``int64`` and ``CrossEntropyLoss`` is used.
        * For binary classification (``task='binary_classification'``), targets are
          cast to ``float32`` in ``{0,1}`` and ``BCEWithLogitsLoss`` is used.
        * For regression, targets are cast to ``float32`` and ``MSELoss`` is used.
    """
    def __init__(
        self,
        num_hidden: int = 1,
        activation: str = "ReLU",
        max_epochs: int = 100,
        optimizer: str = "SGD",
        numerical_index: Optional[List[int]] = None,
        categorical_index: Optional[List[int]] = None,
        num_levels_per_var: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(estimator=None, **kwargs)
        self.num_hidden = num_hidden
        self.activation = activation
        self.max_epochs = max_epochs
        self.optimizer = optimizer

        # Determine output dimension and ensure target dtype compatibility for PyTorch
        self.num_targets = 1
        if self.task == "classification":
            # multiclass -> int64 labels for CrossEntropyLoss
            self.y = self.y.astype(np.int64)
            self.num_targets = int(np.unique(self.y).size)
        elif self.task == "binary_classification":
            # BCEWithLogitsLoss expects float targets in {0,1}
            self.y = self.y.astype(np.float32)
            self.num_targets = 1
        else:  # regression
            self.y = self.y.astype(np.float32)
            self.num_targets = 1

        # Feature indices
        if categorical_index is None:
            # all features are numerical if not specified
            self.categorical_index = None
            self.numerical_index = (
                numerical_index if numerical_index is not None else list(range(self.X.shape[1]))
            )
        else:
            self.categorical_index = categorical_index
            if numerical_index is None:
                # infer numerical indices as the complement
                cat_set = set(categorical_index)
                self.numerical_index = [j for j in range(self.X.shape[1]) if j not in cat_set]
            else:
                # validate disjointness
                overlap = set(numerical_index).intersection(categorical_index)
                if overlap:
                    raise ValueError(f"numerical_index and categorical_index must be disjoint; overlap={sorted(overlap)}")
                self.numerical_index = numerical_index

        self.num_levels_per_var = num_levels_per_var

    def construct_model(
        self, params: Dict
    ) -> Union[NeuralNetRegressor, NeuralNetClassifier, NeuralNetBinaryClassifier]:
        """
        Build a skorch-wrapped :class:`MLP` from the provided hyperparameters.

        Expected keys in ``params``:
            - ``hsize{i}``, ``dropout{i}`` for ``i = 0 .. num_hidden-1``
            - ``lr``: learning rate
            - ``weight_decay``: L2 weight decay
            - ``batch_size``: batch size
            - ``momentum``: (optional) if ``optimizer == 'SGD'``

        Returns:
            A ``skorch.NeuralNet*`` instance ready for ``fit``/``predict``.
        """
        # Extract per-layer widths/dropouts
        try:
            h_sizes = [params[f"hsize{i}"] for i in range(self.num_hidden)]
            dropouts = [params[f"dropout{i}"] for i in range(self.num_hidden)]
        except KeyError as e:
            raise KeyError(
                f"Missing required hyperparameter for layer definition: {e!s}. "
                f"Expected hsize0..hsize{self.num_hidden-1} and dropout0..dropout{self.num_hidden-1}."
            ) from e

        # Choose skorch wrapper + loss
        if self.task == "regression":
            SkNet = NeuralNetRegressor
            criterion = nn.MSELoss
        elif self.task == "classification":  # multiclass
            SkNet = NeuralNetClassifier
            criterion = nn.CrossEntropyLoss
        else:  # binary_classification
            SkNet = NeuralNetBinaryClassifier
            criterion = nn.BCEWithLogitsLoss

        # Build the estimator
        net = SkNet(
            module=MLP,
            module__h_sizes=h_sizes,
            module__dropouts=dropouts,
            module__output_dim=self.num_targets,
            module__activation=self.activation,
            module__numerical_index=self.numerical_index,
            module__categorical_index=self.categorical_index,
            module__num_levels_per_var=self.num_levels_per_var,
            criterion=criterion,
            optimizer=getattr(torch.optim, self.optimizer),
            optimizer__lr=params["lr"],
            optimizer__weight_decay=params["weight_decay"],
            max_epochs=self.max_epochs,
            batch_size=params["batch_size"],
            train_split=ValidSplit(10, stratified=("classification" in self.task)),
            callbacks=[
                EpochScoring(
                    scoring=make_scorer(
                        self.loss_metric,
                        needs_proba=self.needs_proba,
                        greater_is_better=False,  # treat metric as a loss: lower is better
                    ),
                    lower_is_better=True,
                    name="valid_metric",
                ),
                EarlyStopping(patience=15, monitor="valid_metric", load_best=True),
                LRScheduler(
                    policy="ReduceLROnPlateau",
                    monitor="valid_metric",
                    factor=0.1,
                    patience=5,
                    min_lr=1e-5,
                ),
                GradientNormClipping(gradient_limit=5.0),
            ],
            verbose=0,
        )

        # Add momentum if using SGD
        if self.optimizer == "SGD" and "momentum" in params:
            net.set_params(optimizer__momentum=params["momentum"])

        return net

    def get_recommended_configspace(self) -> ConfigurationSpace:
        """
        Recommended hyperparameter search space for the MLP.

        Useful defaults for black-box optimizers (e.g., Optuna, SMAC):

        - Hidden layer sizes: log-uniform integer in ``[8, 256]``
        - Dropout rates: uniform real in ``[0.0, 0.5]``
        - Learning rate: log-uniform real in ``[1e-4, 1e-1]``
        - Weight decay: log-uniform real in ``[1e-8, 1]``
        - Batch size: log-uniform integer in ``[16, 128]``
        - Momentum (SGD only): uniform real in ``[0.5, 0.99]``

        Returns:
            A :class:`ConfigurationSpace` describing the suggested search space.
        """
        config = ConfigurationSpace()

        # Per-layer hyperparameters
        for i in range(self.num_hidden):
            config.add(Integer(f"hsize{i}", lower=8, upper=256, log=True, default=64))
            config.add(Float(f"dropout{i}", lower=0.0, upper=0.5, default=0.1))

        # Optimization hyperparameters
        config.add(Float("lr", lower=1e-4, upper=1e-1, log=True, default=5e-2))
        config.add(Float("weight_decay", lower=1e-8, upper=1.0, log=True, default=1e-2))
        config.add(Integer("batch_size", lower=16, upper=128, log=True, default=32))

        if self.optimizer == "SGD":
            config.add(Float("momentum", lower=0.5, upper=0.99, default=0.9))
            
        return config