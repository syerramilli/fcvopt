import numpy as np
import pandas as pd
import os,joblib
import torch
import random
import optuna

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer

from fcvopt.crossvalidation.sklearn_cvobj import SklearnCVObj
from fcvopt.crossvalidation.optuna_obj import get_optuna_objective
from fcvopt.util.samplers import stratified_sample

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
    estimator=RandomForestClassifier(n_estimators=500, n_jobs=-1),
    X=X,y=y,
    loss_metric=metric,
    needs_proba=True,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    task='binary-classification',
    rng_seed=args.seed
)

#%% 
config = ConfigurationSpace(seed=1234)
config.add([
    Integer('max_depth',bounds=(1,12),log=True),
    Float('min_impurity_decrease',bounds=(1e-8,10),log=True),
    Float('max_features',bounds=(0.005,0.5),log=True),
    Integer('min_samples_split',bounds=(5,250),log=True),
])

config.generate_indices()

#%%

set_seed(args.seed)
config.seed(np.random.randint(2e+4))
init_trials = [dict(conf) for conf in config.latinhypercube_sample(args.n_init)]
start_fold_idxs = stratified_sample(10, args.n_init).tolist()
optuna_obj = get_optuna_objective(cvobj, config, start_fold_idxs, rng_seed=args.seed)

sampler = optuna.samplers.TPESampler(
    n_startup_trials=args.n_init, seed=args.seed
)

remaining_trials = args.n_init + args.n_iter - 1
study_path = os.path.join(save_dir, 'study.pkl')
if os.path.exists(study_path):
    print("Resuming study from", study_path)
    study = joblib.load(study_path)
    remaining_trials = remaining_trials - len(study.trials)
else:
    study = optuna.create_study(
        directions=['minimize'],sampler=sampler,
        study_name=f'rf_{args.dataset}_{args.seed}'
    )

    for trial in init_trials:
        study.enqueue_trial(trial)

if remaining_trials > 0:
    study.optimize(optuna_obj, n_trials=remaining_trials, timeout=None)
else:
    print("Study already reached the target number of trials.")

# save results to file
joblib.dump(study,os.path.join(save_dir,'study.pkl'))