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

from fcvopt.crossvalidation.sklearn_cvobj import SklearnCVObj
from fcvopt.configspace import ConfigurationSpace
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

        if len(h_sizes) == 0:
            raise ValueError("h_sizes must contain at least one hidden layer width.")
        if len(h_sizes) != len(dropouts):
            raise ValueError("dropouts must match the length of h_sizes.")
        if activation not in {"ReLU", "SELU", "Sigmoid"}:
            raise ValueError("activation must be one of {'ReLU','SELU','Sigmoid'}.")

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

        layers: List[nn.Module] = []
        for h, d in zip(h_sizes, dropouts):
            layers.append(nn.Linear(input_dim, h))
            layers.append(getattr(nn, activation)())
            layers.append(nn.AlphaDropout(d) if activation == "SELU" else nn.Dropout(d))
            input_dim = h
        self.hidden_layers = nn.Sequential(*layers)

        self.output = nn.Linear(h_sizes[-1], output_dim)

        if activation == "SELU":
            def init_fn(m: nn.Module):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                    nn.init.zeros_(m.bias)
            self.apply(init_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    Wraps :class:`MLP` in a scikit-learn-compatible skorch estimator.

    Args:
        num_hidden: Number of hidden layers.
        activation: Hidden activation (``'ReLU'``, ``'SELU'``, or ``'Sigmoid'``).
        max_epochs: Maximum training epochs per fold.
        optimizer: Name of PyTorch optimizer (``'SGD'``, ``'Adam'``, or ``'AdamW'``).
        numerical_index: Indices of numerical features.
        categorical_index: Indices of categorical features to embed (optional).
        num_levels_per_var: Cardinalities for each categorical variable.
        **kwargs: Forwarded to :class:`SklearnCVObj`.
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

        self.num_targets = 1
        if self.task == "classification":
            self.y = self.y.astype(np.int64)
            self.num_targets = int(np.unique(self.y).size)
        elif self.task == "binary_classification":
            self.y = self.y.astype(np.float32)
            self.num_targets = 1
        else:
            self.y = self.y.astype(np.float32)
            self.num_targets = 1

        if categorical_index is None:
            self.categorical_index = None
            self.numerical_index = (
                numerical_index if numerical_index is not None else list(range(self.X.shape[1]))
            )
        else:
            self.categorical_index = categorical_index
            if numerical_index is None:
                cat_set = set(categorical_index)
                self.numerical_index = [j for j in range(self.X.shape[1]) if j not in cat_set]
            else:
                overlap = set(numerical_index).intersection(categorical_index)
                if overlap:
                    raise ValueError(
                        f"numerical_index and categorical_index must be disjoint; overlap={sorted(overlap)}"
                    )
                self.numerical_index = numerical_index

        self.num_levels_per_var = num_levels_per_var

    def construct_model(
        self, params: Dict
    ) -> Union[NeuralNetRegressor, NeuralNetClassifier, NeuralNetBinaryClassifier]:
        try:
            h_sizes = [params[f"hsize{i}"] for i in range(self.num_hidden)]
            dropouts = [params[f"dropout{i}"] for i in range(self.num_hidden)]
        except KeyError as e:
            raise KeyError(
                f"Missing required hyperparameter: {e!s}. "
                f"Expected hsize0..hsize{self.num_hidden-1} and dropout0..dropout{self.num_hidden-1}."
            ) from e

        if self.task == "regression":
            SkNet = NeuralNetRegressor
            criterion = nn.MSELoss
        elif self.task == "classification":
            SkNet = NeuralNetClassifier
            criterion = nn.CrossEntropyLoss
        else:
            SkNet = NeuralNetBinaryClassifier
            criterion = nn.BCEWithLogitsLoss

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
                        response_method='predict_proba' if self.needs_proba else 'predict',
                        greater_is_better=False,
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

        if self.optimizer == "SGD" and "momentum" in params:
            net.set_params(optimizer__momentum=params["momentum"])

        return net

    def get_recommended_configspace(self) -> ConfigurationSpace:
        config = ConfigurationSpace()

        for i in range(self.num_hidden):
            config.add(Integer(f"hsize{i}", lower=8, upper=256, log=True, default=64))
            config.add(Float(f"dropout{i}", lower=0.0, upper=0.5, default=0.1))

        config.add(Float("lr", lower=1e-4, upper=1e-1, log=True, default=5e-2))
        config.add(Float("weight_decay", lower=1e-8, upper=1.0, log=True, default=1e-2))
        config.add(Integer("batch_size", lower=16, upper=128, log=True, default=32))

        if self.optimizer == "SGD":
            config.add(Float("momentum", lower=0.5, upper=0.99, default=0.9))

        return config
