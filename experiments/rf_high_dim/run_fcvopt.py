import numpy as np
import pandas as pd
import os,joblib
import torch
import random

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer

from fcvopt.optimizers.fcvopt import FCVOpt
from fcvopt.optimizers.mtbo_cv import MTBOCVOpt
from fcvopt.crossvalidation.sklearn_cvobj import SklearnCVObj
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_openml


from argparse import ArgumentParser

parser = ArgumentParser(description='RF classification')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument(
    '--acq',
    type=str,required=True,
    choices=['lcb','kg','lcb_batch','kg_batch','mtbo']
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
    'gina':1038,
    'hiva':1039,
    'madelon':1485,
    'bioresponse':4134,
}


X,y = fetch_openml(data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=False, parser='auto')

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


#%% parse acquistion arguments
acq_args = {}
acq_args['acq_function'] = 'LCB' if 'lcb' in args.acq else 'KG'
if 'batch' in args.acq:
    acq_args['batch_acquisition']=True
    acq_args['acquisition_q']=4

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
if args.acq == 'mtbo':
    opt = MTBOCVOpt(
        obj=cvobj.cvloss,
        n_folds=cvobj.cv.get_n_splits(),
        fold_initialization='stratified',
        config=config,
        verbose=1,
        n_jobs=-1
    )
else:
    acq_args = {}
    acq_args['acq_function'] = 'LCB' if 'lcb' in args.acq else 'KG'
    if 'batch' in args.acq:
        acq_args['batch_acquisition']=True
        acq_args['acquisition_q']=4

    opt = FCVOpt(
        obj=cvobj.cvloss,
        n_folds=cvobj.cv.get_n_splits(),
        n_repeats=1,
        fold_selection_criterion='variance_reduction',
        fold_initialization='stratified',
        config=config,
        verbose=1,
        n_jobs=-1,
        **acq_args
    )

out = opt.run(args.n_iter, n_init=args.n_init)