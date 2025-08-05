import numpy as np
import torch
import torch.nn as nn
try:
    from skorch import NeuralNetRegressor, NeuralNetClassifier, NeuralNetBinaryClassifier
    from skorch.callbacks import GradientNormClipping, EarlyStopping, EpochScoring, LRScheduler
    from skorch.dataset import ValidSplit
except:
    raise ImportError('skorch must be installed to use the MLPCVObj class')
from sklearn.metrics import make_scorer
from typing import List, Optional, Dict, Union

from .sklearn_cvobj import SklearnCVObj
from ..configspace import ConfigurationSpace
from ConfigSpace import Float, Integer

class MLP(nn.Module):
    """Feed-forward neural network with embeddings for categorical inputs and dropout.

    This module supports:
      - purely numerical inputs
      - mixed numerical and categorical inputs (via 2-dimensional embeddings)
      - any choice of ReLU, SELU (with α-dropout), or Sigmoid activations
      - multiple hidden layers with specific sizes and dropout rates

    Args:
        h_sizes: Hidden‐layer sizes. Length determines number of layers; at least one.
        dropouts: Dropout rate per hidden layer. Must match length of `h_sizes`.
        output_dim: Dimension of final output. Use 1 for regression/binary classification,
            or number of classes for multiclass.
        numerical_index: Indices of numerical features in the input vector.
        activation: One of {'ReLU', 'SELU', 'Sigmoid'}. If 'SELU', weights are
            initialized for self-normalizing nets and α-dropout is used.
            Defaults to 'Sigmoid'.
        categorical_index: Indices of categorical features to embed. If None, no embeddings are used.
            Defaults to None.
        num_levels_per_var: Number of categories for each index in `categorical_index`. Must
            align in length. Defaults to None.
    """
    def __init__(
        self,
        h_sizes: List[int],
        dropouts: List[float],
        output_dim: int,
        numerical_index: List[int],
        activation: str = 'Sigmoid',
        categorical_index: Optional[List[int]] = None,
        num_levels_per_var: Optional[List[int]] = None
    ):
        super().__init__()

        # Embedding setup for categorical variables
        if categorical_index is None:
            input_dim = len(numerical_index)
            self.embedding_layers = None
        else:
            input_dim = len(numerical_index) + 2 * len(categorical_index)
            self.embedding_layers = nn.ModuleList([
                nn.Embedding(levels, 2)
                for levels in num_levels_per_var
            ])
            self.categorical_index = torch.tensor(categorical_index, dtype=torch.long)

        self.numerical_index = torch.tensor(numerical_index, dtype=torch.long)

        # Build hidden layers
        layers: List[nn.Module] = []
        for h, d in zip(h_sizes, dropouts):
            layers.append(nn.Linear(input_dim, h))
            layers.append(getattr(nn, activation)())
            if activation == 'SELU':
                layers.append(nn.AlphaDropout(d))
            else:
                layers.append(nn.Dropout(d))
            input_dim = h
        self.hidden_layers = nn.Sequential(*layers)

        # Final output layer
        self.output = nn.Linear(h_sizes[-1], output_dim)

        # SELU initialization
        if activation == 'SELU':
            def init_fn(m: nn.Module):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                    nn.init.zeros_(m.bias)
            self.apply(init_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle embeddings if present
        if self.embedding_layers is not None:
            embeds = []
            for idx, emb in enumerate(self.embedding_layers):
                idx_col = self.categorical_index[idx]
                embeds.append(emb(x[..., idx_col].long()))
            x_num = x[..., self.numerical_index]
            x = torch.cat([torch.cat(embeds, dim=1), x_num], dim=1)

        x = self.hidden_layers(x)
        return self.output(x)


class MLPCVObj(SklearnCVObj):
    """Cross-validation objective for a feed forward neural network implemented with PyTorch and skorch.

    Wraps :class:`MLP` in a scikit-learn-compatible estimator, then computes the 
    K-fold CV loss via :class:`fcvopt.crossvalidation.SklearnCVObj`. 
    
    Includes the following built-in features:
    - **Early stopping**: stops training when the validation loss does not improve for 15 epochs.
    - **Learning-rate scheduling**: reduces LR by a factor of 0.1 if validation loss plateaus for 5 epochs (min LR = 1e-5).
    - **Gradient norm clipping**: caps gradient norm at 5.0 each update.

    Args:
        num_hidden: Number of hidden layers. Each layers uses different size and dropout 
            hyperparameters. These are expected to be provided in the `params` dictionary 
            as `hsize{i}` and `dropout{i}` for `i` in [0, ..., num_hidden-1].
            Defaults to 1.
        activation: Activation for hidden layers: 'ReLU', 'SELU', or 'Sigmoid'.
            Defaults to 'ReLU'.
        max_epochs: Maximum training epochs per fold. Defaults to 100.
        optimizer: Name of PyTorch optimizer: 'SGD', 'Adam', or 'RMSprop'.
            Defaults to 'SGD'.
        numerical_index: Indices of numerical features. If None, all features are treated
            numerically. Defaults to None.
        categorical_index: Indices of categorical features. Defaults to None.
        num_levels_per_var: Cardinalities of each categorical variable. Defaults to None.
        **kwargs: Passed through to :class:`SklearnCVObj` (e.g. `estimator`, `X`, `y`, `task`,
            `loss_metric`, CV settings, parallel jobs, etc.).
    """
    def __init__(
        self,
        num_hidden: int = 1,
        activation: str = 'ReLU',
        max_epochs: int = 100,
        optimizer: str = 'SGD',
        numerical_index: Optional[List[int]] = None,
        categorical_index: Optional[List[int]] = None,
        num_levels_per_var: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(estimator=None, **kwargs)
        self.num_hidden = num_hidden
        self.activation = activation
        self.max_epochs = max_epochs
        self.optimizer = optimizer

        # Determine output dimension
        self.num_targets = 1
        if 'classification' in self.task:
            self.y = self.y.astype(np.float32)
            if self.task == 'multiclass_classification':
                self.num_targets = int(np.unique(self.y).size)

        # Feature indices
        self.categorical_index = categorical_index
        self.numerical_index = (
            numerical_index
            if numerical_index is not None
            else list(range(self.X.shape[1]))
        )
        self.num_levels_per_var = num_levels_per_var

    def construct_model(self, params: Dict) -> Union[NeuralNetRegressor, NeuralNetClassifier, NeuralNetBinaryClassifier]:
        """Builds a Skorch-wrapped MLP with given hyperparameters.

        Args:
            params (Dict): Hyperparameter mapping, expected to contain keys:
                - 'hsize0', 'dropout0', ..., up to `num_hidden`
                - 'lr': the learning rate,
                - 'weight_decay': the L2 weight decay
                - 'batch_size': the batch size for training
                - if `self.optimizer == 'SGD'`, then 'momentum' is an optional
                hyperparameter.

        Returns:
            A `skorch.NeuralNet*` instance ready for `.fit()`/.predict().
        """
        # Extract layer sizes and dropouts
        h_sizes = [params[f'hsize{i}'] for i in range(self.num_hidden)]
        dropouts = [params[f'dropout{i}'] for i in range(self.num_hidden)]

        # Choose the right Skorch class & loss
        if self.task == 'regression':
            SkNet = NeuralNetRegressor
            criterion = nn.MSELoss
        elif self.task == 'multiclass_classification':
            SkNet = NeuralNetClassifier
            criterion = nn.CrossEntropyLoss
        else:
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
            optimizer__lr=params['lr'],
            optimizer__weight_decay=params['weight_decay'],
            max_epochs=self.max_epochs,
            batch_size=params['batch_size'],
            train_split=ValidSplit(
                10,
                stratified=('classification' in self.task)
            ),
            callbacks=[
                EpochScoring(
                    scoring=make_scorer(self.loss_metric, needs_proba=self.needs_proba),
                    lower_is_better=True,
                    name='valid_metric'
                ),
                EarlyStopping(patience=15, monitor='valid_metric', load_best=True),
                LRScheduler(
                    policy='ReduceLROnPlateau',
                    monitor='valid_metric',
                    factor=0.1,
                    patience=5,
                    min_lr=1e-5
                ),
                GradientNormClipping(gradient_limit=5.0)
            ],
            verbose=0
        )

        # Add momentum if using SGD
        if self.optimizer == 'SGD' and 'momentum' in params:
            net.set_params(optimizer__momentum=params['momentum'])

        return net
    
    def get_recommended_configspace(self) -> ConfigurationSpace:
        '''Returns a recommended hyperparameter search space for the feed-forward neural network.
        This space can be plugged into the optimizers in `fcvopt` for hyperparameter tuning.

        The search space includes:
            - Hidden layer sizes (log-uniform between 8 and 256)
            - Dropout rates (uniform between 0.0 and 0.5)
            - Learning rate (log-uniform between 1e-4 and 0.1)
            - Weight decay (log-uniform between 1e-8 and 1)
            - Batch size (log-uniform between 16 and 128)
            - Momentum (if using SGD, uniform between 0.5 and 0.99)
        '''
        config = ConfigurationSpace()

        # Add hyperparameters for hidden layers
        for i in range(self.num_hidden):
            config.add(
                Integer(f'hsize{i}', lower=8, upper=256, log=True, default=64)
            )
            config.add(
                Float(f'dropout{i}', lower=0.0, upper=0.5, default=0.1)
            )
        
        # optimization hyperparameters
        config.add(
            Float('lr', lower=1e-4, upper=0.1, log=True, default=0.05), # learning rate
            Float('weight_decay', lower=1e-8, upper=1, log=True, default=0.01), # L2 weight decay
            Integer('batch_size', lower=16, upper=128, log=True, default=32) # batch size
        )

        if self.optimizer == 'SGD':
            config.add(
                Float('momentum', lower=0.5, upper=0.99, default=0.9) # momentum for SGD
            )

        config.generate_indices()
        return config