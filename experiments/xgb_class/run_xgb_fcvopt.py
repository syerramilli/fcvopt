import numpy as np
import pandas as pd
import os,joblib
import torch
import random

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer,Categorical

from fcvopt.optimizers.fcvopt import FCVOpt
from fcvopt.crossvalidation.sklearn_cvobj import XGBoostCVObjEarlyStopping
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from sklearn.datasets import fetch_openml

from argparse import ArgumentParser

parser = ArgumentParser(description='XGBoost classification')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument(
    '--acq',
    type=str,required=True,
    choices=['lcb','kg','lcb_batch','kg_batch']
)
parser.add_argument('--n_init',type=int,required=True)
parser.add_argument('--n_iter',type=int,required=True)
parser.add_argument('--n_folds',type=int,default=10)
parser.add_argument('--n_repeats',type=int,default=1)
parser.add_argument('--seed',type=int,default=123)
args = parser.parse_args()

save_dir = os.path.join(args.save_dir,args.dataset,args.acq,'seed_%d'%args.seed)
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
        n_estimators=2000,tree_method='approx',enable_categorical=True
    ),
    X=X,y=y,
    loss_metric=metric,
    needs_proba=True,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    task='binary-classification',
    early_stopping_rounds=50
)


#%% parse acquistion arguments
acq_args = {}
acq_args['acq_function'] = 'LCB' if 'lcb' in args.acq else 'KG'
if 'batch' in args.acq:
    acq_args['batch_acquisition']=True
    acq_args['acquisition_q']=4

#%% 
config = ConfigurationSpace(seed=1234)
config.add_hyperparameters([
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
opt = FCVOpt(
    obj=cvobj.cvloss,
    n_folds=cvobj.cv.get_n_splits(),
    n_repeats=1,
    fold_selection_criterion='variance_reduction',
    fold_initialization='stratified',
    config=config,
    save_iter=10,
    save_dir = save_dir,
    verbose=2,
    **acq_args
)

out = opt.run(args.n_iter,n_init=args.n_init)
# save to disk
opt.save_to_file(save_dir)