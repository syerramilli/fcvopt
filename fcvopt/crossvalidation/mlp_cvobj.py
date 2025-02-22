import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor,NeuralNetClassifier,NeuralNetBinaryClassifier
from skorch.callbacks import GradientNormClipping, EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import ValidSplit
from sklearn.metrics import make_scorer
from typing import List, Tuple, Optional, Dict

from ..crossvalidation.sklearn_cvobj import SklearnCVObj

class MLP(nn.Module):
    '''A feed forward neural network with dropout regularization for tabular data

    Args:
        h_sizes: a list of hidden layer sizes. The length of this list determines the
            number of hidden layers. Must have atleast one element.
        dropouts: a list of dropout rates for each hidden layer. Must have the same length
            as `h_sizes`.
        output_dim: the output dimension of the network. For regression and binary classification,
            this is 1. For multiclass classification, this is the number of classes.
        numerical_index: a list of indices of the numerical features in the input data. If None,
            all features are assumed to be numerical.
        activation: the activation function to use for the hidden layers. Must be one of 'ReLU',
            'SELU', or 'Sigmoid'. If 'SELU', the network is initialized using the method described
            in https://arxiv.org/abs/1706.02515, and the dropout layers are replaced with alpha
            dropout layers. (default: 'Sigmoid')
        categorical_index: a list of indices of the categorical features in the input data. If not 
            None, the categorical features are first embedded using an embedding layer. If None, all
            features are assumed to be numerical. (default: None)
        num_levels_per_var: a list of the number of levels for each categorical variable. Must have
            the same length as `categorical_index`. (default: None)
    '''
    def __init__(
        self,
        h_sizes:List[int],
        dropouts:List[float],
        output_dim:int,
        numerical_index:int,
        activation:str='Sigmoid',
        categorical_index:Optional[List[int]]=None,
        num_levels_per_var:Optional[List[int]]=None
    ):
        super(MLP,self).__init__()

        if categorical_index is None:
            input_dim = len(numerical_index)
            self.register_buffer('categorical_index',categorical_index)
        else:
            input_dim = 2*len(categorical_index) + len(numerical_index)
            self.register_buffer('categorical_index',torch.Tensor(categorical_index).long())
            self.embedding_layers = nn.ModuleList([
                nn.Embedding(num_levels,2) for num_levels in num_levels_per_var
            ])
        
        self.register_buffer('numerical_index',torch.tensor(numerical_index).long())

        hidden_layers = []
        for hsize,droprate in zip(h_sizes,dropouts):
            hidden_layers.append(nn.Linear(input_dim,hsize))
            hidden_layers.append(getattr(nn,activation)())
            hidden_layers.append(nn.AlphaDropout(droprate) if activation == 'SELU' else nn.Dropout(droprate))
            input_dim = hsize
        
        self.hidden_layers = nn.Sequential(*hidden_layers)  
        self.output = nn.Linear(h_sizes[-1],output_dim)

        if activation == 'SELU':
            # Ensure correct initialization
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight.data, mode='fan_in', nonlinearity='linear'
                    )
                    nn.init.zeros_(m.bias.data)

            self.apply(init_weights)


    def forward(self,x):
        if self.categorical_index is not None:
            embeddings = []
            for i,e in enumerate(self.embedding_layers):
                embeddings.append(e(x[...,self.categorical_index[i]].long()))
            
            embeddings = torch.cat(embeddings,1)
            x = torch.cat([embeddings,x[...,self.numerical_index]],1)

        x = self.hidden_layers(x)    
        return self.output(x)

class MLPCVObj(SklearnCVObj):
    '''A cross-validation object for a feed forward neural network with dropout regularization
    for tabular data.

    The number of hidden layers, given by `num_hidden`, is fixed. However, the size of each hidden
    layer and the corresponding dropout rate are expected to be hyperparameters. 

    :note: The network is constructed using the `skorch` library, that wraps PyTorch modules
        into scikit-learn compatible estimators. The `skorch` library is not a dependency of
        `fcvopt` and must be installed separately.

    Args:
        num_hidden: number of hidden layers (default: 1)
        activation: the activation function to use for the hidden layers. Must be one of 'ReLU',
            'SELU', or 'Sigmoid'. If 'SELU', the network is initialized using the method described
            in https://arxiv.org/abs/1706.02515, and the dropout layers are replaced with alpha
            dropout layers. (default: 'Sigmoid')
        max_epochs: maximum number of epochs for training (default: 100)
        optimizer: the optimizer to use for training. Must be one of 'SGD', 'Adam', or 'RMSprop'.
            (default: 'SGD')
        numerical_index: a list of indices of the numerical features in the input data. If None,
            all features are assumed to be numerical.
        categorical_index: a list of indices of the categorical features in the input data. If not 
            None, the categorical features are first embedded using an embedding layer. If None, all
            features are assumed to be numerical. (default: None)
        num_levels_per_var: a list of the number of levels for each categorical variable. Must have
            the same length as `categorical_index`. (default: None)
        kwargs: additional keyword arguments to pass to the `SklearnCVObj` constructor. Do not pass
            the `estimator` argument here.
    '''
    def __init__(
        self,
        num_hidden:int=1,
        activation:str='ReLU',
        max_epochs:int=100,
        optimizer:int='SGD',
        numerical_index=None,
        categorical_index=None,
        num_levels_per_var=None,
        **kwargs
    ):
        super().__init__(estimator=None,**kwargs)
        self.num_hidden = num_hidden
        self.activation = activation
        self.max_epochs=max_epochs
        self.optimizer=optimizer

        self.num_targets = 1 # for regression and binary-classification
        if 'classification' in self.task:
            self.y = self.y.astype(np.float32)
            if self.task =='classification':
                self.num_targets = np.unique(self.y).size
            

        self.categorical_index = categorical_index
        self.num_levels_per_var = num_levels_per_var
        if self.categorical_index is None:
            self.numerical_index = np.arange(self.X.shape[1]).tolist()
        else:
            self.numerical_index = numerical_index
    
    def construct_model(self, params):
        # this is specific to the model I construct
        h_sizes = [
            params['hsize%d'%i] for i in range(self.num_hidden) 
        ]
        dropouts = [
            params['dropout%d'%i] for i in range(self.num_hidden) 
        ]
        
        # param_groups = [
        #     ('hidden*.%d.weight'%i,{'weight_decay':params['lam_weights']}) \
        #     for i in range(self.num_hidden)
        # ]
        
        # param_groups.append(
        #     ('output.weight',{'weight_decay':params['lam_weights']}),
        # )

        # if self.categorical_index is not None:
        #     param_groups.append(
        #         ('embedding*.weight',{'weight_decay':params['lam_embedding']})
        #     )
        
        if self.task == 'regression':
            skorch_class = NeuralNetRegressor
            criterion = nn.MSELoss
        elif self.task == 'classification':
            skorch_class = NeuralNetClassifier
            criterion = nn.CrossEntropyLoss
        else:
            skorch_class = NeuralNetBinaryClassifier
            criterion = nn.BCEWithLogitsLoss

        model = skorch_class(
            module = MLP,
            criterion=criterion,
            module__h_sizes=h_sizes,
            module__dropouts=dropouts,
            module__output_dim=self.num_targets,
            module__activation=self.activation,
            module__numerical_index=self.numerical_index,
            module__categorical_index=self.categorical_index,
            module__num_levels_per_var=self.num_levels_per_var,
            callbacks=[
                EpochScoring(
                    scoring = make_scorer(self.loss_metric,needs_proba=self.needs_proba),
                    lower_is_better=True,
                    name='valid_metric'
                ),
                EarlyStopping(patience=15, monitor='valid_metric',load_best=True),
                LRScheduler(
                    policy='ReduceLROnPlateau',monitor='valid_metric',
                    factor=0.1,
                    mode='min',
                    patience=5,
                    verbose=False,
                    min_lr=1e-5
                )
            ],
            optimizer=getattr(torch.optim,self.optimizer),
            optimizer__lr = params['lr'],
            optimizer__weight_decay = params['weight_decay'],
            max_epochs=self.max_epochs,
            batch_size=params['batch_size'],
            train_split=ValidSplit(10,stratified=True if 'classification' in self.task else False),
            verbose=0
        )

        if self.optimizer == 'SGD':
            model.set_params(optimizer__momentum=params['momentum'])
        
        return model