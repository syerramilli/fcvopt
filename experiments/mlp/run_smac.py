import numpy as np
import pandas as pd
import os,joblib
import torch
import random

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer

from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import AbstractInitialDesign

from fcvopt.crossvalidation.mlp_cvobj import MLPCVObj
from sklearn.metrics import mean_squared_error

#from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_openml

from argparse import ArgumentParser

parser = ArgumentParser(description='MLP regression')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--n_init',type=int,required=True)
parser.add_argument('--n_iter',type=int,required=True)
parser.add_argument('--n_folds',type=int,default=10)
parser.add_argument('--n_repeats',type=int,default=1)
parser.add_argument('--seed',type=int,default=123)
args = parser.parse_args()

save_dir = os.path.join(args.save_dir,args.dataset,'SMAC','seed_%d'%args.seed)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#%%
def metric(y_true,y_pred):
    # rrmse
    return np.sqrt(mean_squared_error(y_true,y_pred))/np.std(y_true,ddof=0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#%% fetch dataset
DATA_IDS = {
   'house_16H':44139,
   'house_sales':44144,
   'superconduct':44148,
   'pol':44133
}

X,y = fetch_openml(data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=False,parser='auto')

#%% define estimator and cross-validation objective
set_seed(1)
n_hidden = 2
cvobj = MLPCVObj(
    num_hidden=n_hidden,
    activation='SELU',
    max_epochs=100,
    optimizer='AdamW',
    task='regression',
    X=X.astype(np.float32),y=y.reshape(-1,1).astype(np.float32),
    loss_metric=metric,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    scale_output=True,
    input_preprocessor=StandardScaler()
)

#%% 
config = ConfigurationSpace(seed=1234)
config.add_hyperparameters([
    Integer('hsize%d'%i,bounds=(8,256),log=True) for i in range(n_hidden)
])

# dropouts
config.add_hyperparameters([
    Float('dropout%d'%i,bounds=(0,0.5)) for i in range(n_hidden)
])

# remaining hyperparameters
config.add_hyperparameters([
    Float('lr',bounds=(1e-5,0.1),log=True),
    Integer('batch_size',bounds=(32,2048),log=True),
    Float('weight_decay',bounds=(1e-8,1),log=True),
    Float('callbacks__LRScheduler__factor',bounds=(0.1,0.5))
])
config.generate_indices()

#%%
def cvloss(config,seed:int=0) -> float:
    rng  = np.random.default_rng(seed=seed)
    fold_idxs = rng.choice(len(cvobj.train_test_splits))
    
    return cvobj.cvloss(params=config.get_dictionary(),fold_idxs=[fold_idxs])

set_seed(args.seed)
config.seed(np.random.randint(2e+4))
initial_confs = config.latinhypercube_sample(args.n_init)

scenario = Scenario(
    config,
    n_trials=args.n_init+args.n_iter-1,  
    output_directory=save_dir,
    deterministic=False
)

initial_design = AbstractInitialDesign(
    scenario=scenario,n_configs=0,additional_configs=initial_confs)

# Create our SMAC object and pass the scenario and the train method
smac = HyperparameterOptimizationFacade(
    scenario,
    cvloss,
    initial_design=initial_design,
    overwrite=True,
)

incumbent = smac.optimize()