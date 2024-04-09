import numpy as np
import pandas as pd
import os,joblib
import torch
import random
import optuna

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer,Categorical

from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import AbstractInitialDesign

from fcvopt.crossvalidation.sklearn_cvobj import SklearnCVObj
from fcvopt.crossvalidation.optuna_obj import get_optuna_objective
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_openml

from argparse import ArgumentParser

parser = ArgumentParser(description='RF classification')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--n_init',type=int,required=True)
parser.add_argument('--n_iter',type=int,required=True)
parser.add_argument('--n_folds',type=int,default=10)
parser.add_argument('--n_repeats',type=int,default=1)
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

save_dir = os.path.join(args.save_dir,args.dataset,'optuna','seed_%d'%args.seed)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not args.verbose:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

#%%
def metric(y_true,y_pred):
    return np.sqrt(1-roc_auc_score(y_true,y_pred))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#%% fetch dataset
DATA_IDS = {
    'gina':1038,
    'hiva':1039,
    'madelon':1485,
    'bioresponse':4134,
}

X,y = fetch_openml(data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=True,parser='auto')

#%% define estimator and cross-validation objective
set_seed(1)
cvobj = SklearnCVObj(
    estimator=RandomForestClassifier(n_estimators=500),
    X=X,y=y,
    loss_metric=metric,
    needs_proba=True,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    task='binary-classification',
)

#%% 
config = ConfigurationSpace(seed=1234)
config.add_hyperparameters([
    Integer('max_depth',bounds=(1,12),log=True),
    Float('min_impurity_decrease',bounds=(1e-8,10),log=True),
    Float('max_features',bounds=(0.005,0.5),log=True),
    Integer('min_samples_split',bounds=(5,250),log=True),
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
    study_name=f'rf_{args.dataset}_{args.seed}'
)

for trial in init_trials:
    study.enqueue_trial(trial)

study.optimize(optuna_obj, n_trials=args.n_init+args.n_iter-1, timeout=None)

# save results to file
joblib.dump(study,os.path.join(save_dir,'study.pkl'))