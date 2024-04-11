import numpy as np
import pandas as pd
import os,joblib
import torch
import random
import optuna

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float, Integer, Categorical

from fcvopt.crossvalidation.optuna_obj import get_optuna_objective
from fcvopt.crossvalidation.resnet_cvobj import ResNetCVObj
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import QuantileTransformer

from sklearn.datasets import fetch_openml

from argparse import ArgumentParser

parser = ArgumentParser(description='Tabular classification')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--n_init',type=int,required=True)
parser.add_argument('--n_iter',type=int,required=True)
parser.add_argument('--n_folds',type=int,default=10)
parser.add_argument('--n_repeats',type=int,default=1)
parser.add_argument('--seed',type=int,default=123)
args = parser.parse_args()

save_dir = os.path.join(args.save_dir,args.dataset,'optuna','seed_%d'%args.seed)
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
   'pol':44133,
   'bike_sharing':44142,
   'california':44025
}

X,y = fetch_openml(data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=False,parser='auto')

#%% define estimator and cross-validation objective
set_seed(1)
cvobj = ResNetCVObj(
    max_epochs=100,
    optimizer='AdamW',
    task='regression',
    X=X.astype(np.float32),y=y.reshape(-1,1).astype(np.float32),
    loss_metric=metric,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    scale_output=True,
    input_preprocessor=QuantileTransformer(output_distribution='uniform')
)

#%% 
config = ConfigurationSpace()

config.add_hyperparameters([
    Integer('n_hidden', bounds=(1, 6), log=True),
    Integer('layer_size', bounds=(8, 512), log=True),
    Categorical('normalization', ['batchnorm', 'layernorm']),
    Float('hidden_factor', bounds=(1, 4)),
    Float('hidden_dropout', bounds=(0, 0.5)),
    Float('residual_dropout', bounds=(0, 0.5)),
    Float('lr', bounds=(1e-5, 0.1), log=True),
    Float('weight_decay', bounds=(1e-8, 1e-2), log=True),
])
config.generate_indices()

#%%
optuna_obj = get_optuna_objective(cvobj, config)

set_seed(args.seed)
config.seed(np.random.randint(2e+4))
init_trials = [conf.get_dictionary() for conf in config.latinhypercube_sample(args.n_init)]

sampler = optuna.samplers.TPESampler(
    n_startup_trials=args.n_init
)

study = optuna.create_study(
    directions=['minimize'],sampler=sampler,
    study_name=f'tab_resnet_{args.dataset}_{args.seed}'
)

n_trials = args.n_init+args.n_iter-1
if len(study.trials) == 0:
    # optuna run beginning from scratch

    # initialize trials
    for trial in init_trials:
        study.enqueue_trial(trial)
    
else:
    # resuming interrupted study
    n_trials = n_trials - len(study.trials)

study.optimize(optuna_obj, n_trials=n_trials, timeout=None)

# save results to file
joblib.dump(study,os.path.join(save_dir,'study.pkl'))