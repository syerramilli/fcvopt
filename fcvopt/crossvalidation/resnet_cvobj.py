import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor,NeuralNetClassifier,NeuralNetBinaryClassifier
from skorch.callbacks import GradientNormClipping, EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import ValidSplit
from sklearn.metrics import make_scorer
from typing import List, Tuple, Optional, Dict

from ..crossvalidation.sklearn_cvobj import SklearnCVObj

def make_normalization(normalization:str, input_dim:int):
    '''utility function to return the normlaiation layer'''
    return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
        normalization
    ](input_dim)

class ResNetBlock(nn.Module):
    def __init__(
        self, 
        input_dim:int, 
        normalization:str,
        hidden_factor:float=2, 
        hidden_dropout:float = 0.1, 
        residual_dropout:float = 0.05
    ):
        super().__init__()
        # hidden size
        d_hidden = int(hidden_factor *  input_dim)
        
        self.ff = nn.Sequential(
            make_normalization(normalization, input_dim),
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(hidden_dropout), # hidden dropout
            nn.Linear(d_hidden, input_dim),
            nn.Dropout(residual_dropout) # residual dropset
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x + self.ff(x)

class TabularResNet(nn.Module):
    def __init__(
        self, 
        input_dim:int,
        output_dim:int, 
        n_hidden:int=2,
        layer_size:int=64,
        normalization:str='batchnorm',
        hidden_factor:float=2.,
        hidden_dropout:float=0.1,
        residual_dropout:float=0.05
    ):
        super(TabularResNet, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(input_dim, layer_size)
        )
        for _ in range(n_hidden):
            self.ff.append(ResNetBlock(layer_size, normalization, hidden_factor, hidden_dropout, residual_dropout))
        
        # output layer
        self.prediction = nn.Sequential(
            make_normalization(normalization, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, output_dim)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.prediction(self.ff(x))


class ResNetCVObj(SklearnCVObj):
    def __init__(
        self,
        max_epochs:int=100,
        optimizer:int='SGD',
        **kwargs
    ):
        super().__init__(estimator=None,**kwargs)
        self.max_epochs=max_epochs
        self.optimizer=optimizer

        self.num_targets = 1 # for regression and binary-classification
        if 'classification' in self.task:
            self.y = self.y.astype(np.float32)
            if self.task =='classification':
                self.num_targets = np.unique(self.y).size
            

        self.input_dim = self.X.shape[1]
    
    def construct_model(self, params):
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
            module = TabularResNet,
            criterion=criterion,
            module__input_dim=self.input_dim,
            module__output_dim=self.num_targets,
            module__n_hidden= params['n_hidden'],
            module__layer_size= params['layer_size'],
            module__hidden_factor= params['hidden_factor'],
            module__hidden_dropout= params['hidden_dropout'],
            module__residual_dropout= params['residual_dropout'],
            callbacks=[
                EpochScoring(
                    scoring = make_scorer(self.loss_metric, needs_proba=self.needs_proba),
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
            batch_size=params.get('batch_size', 256),
            train_split=ValidSplit(10,stratified=True if 'classification' in self.task else False),
            verbose=0
        )

        if self.optimizer == 'SGD':
            model.set_params(optimizer__momentum=params['momentum'])
        
        return model