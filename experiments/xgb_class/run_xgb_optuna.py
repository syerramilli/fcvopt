import numpy as np
import pandas as pd
import os,joblib
import torch
import random
import optuna

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer,Categorical

from fcvopt.crossvalidation.sklearn_cvobj import XGBoostCVObjEarlyStopping
from fcvopt.crossvalidation.optuna_obj import get_optuna_objective
from fcvopt.util.samplers import stratified_sample

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from sklearn.datasets import fetch_openml

from argparse import ArgumentParser

parser = ArgumentParser(description='XGBoost classification')
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
    return np.sqrt(1-roc_auc_score(y_true,y_pred))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#%% fetch dataset
DATA_IDS = {
    'compass':44162,
    'eye_movements':44157,
    'electricity':44156
}

X,y = fetch_openml(data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=True)

#%% define estimator and cross-validation objective
set_seed(1)
cvobj = XGBoostCVObjEarlyStopping(
    estimator=XGBClassifier(
        n_estimators=2000,tree_method='approx',enable_categorical=True,n_jobs=-1
    ),
    X=X,y=y,
    loss_metric=metric,
    needs_proba=True,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    task='binary-classification',
    early_stopping_rounds=50,
    rng_seed=args.seed
)

#%% 
config = ConfigurationSpace(seed=1234)
config.add([
    Float('learning_rate',bounds=(1e-5,0.95),log=True),
    Integer('max_depth',bounds=(1,12),log=True),
    Integer('max_leaves',bounds=(2,1024),log=True),
    Float('reg_alpha',bounds=(1e-8,100),log=True),
    Float('reg_lambda',bounds=(1e-8,100),log=True),
    Float('gamma',bounds=(1e-8,100),log=True),
    Float('subsample',bounds=(0.1,1.)),
    Float('colsample_bytree',bounds=(0.1,1.)),
    Categorical('grow_policy', ['depthwise','lossguide'])
])
config.generate_indices()

#%%
set_seed(args.seed)
config.seed(np.random.randint(2e+4))
init_trials = [dict(conf) for conf in config.latinhypercube_sample(args.n_init)]
start_fold_idxs = stratified_sample(10, args.n_init).tolist()
optuna_obj = get_optuna_objective(cvobj, config, start_fold_idxs)

sampler = optuna.samplers.TPESampler(
    n_startup_trials=args.n_init, seed=args.seed
)

# Calculate remaining trials: total desired minus already completed trials.
remaining_trials = args.n_init + args.n_iter - 1
study_path = os.path.join(save_dir, 'study.pkl')
if os.path.exists(study_path):
    print("Resuming study from", study_path)
    study = joblib.load(study_path)
    remaining_trials = remaining_trials - len(study.trials)
else:
    study = optuna.create_study(
        directions=['minimize'],
        sampler=sampler,
        study_name=f'xgb_{args.dataset}_{args.seed}'
    )
    for trial in init_trials:
        study.enqueue_trial(trial)

if remaining_trials > 0:
    study.optimize(optuna_obj, n_trials=remaining_trials, timeout=None)
else:
    print("Study already reached the target number of trials.")

# Save (or update) the study file
joblib.dump(study, study_path)
