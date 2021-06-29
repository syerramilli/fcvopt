import numpy as np
import pandas as pd
import os,joblib
import torch
import random

from fcvopt.configspace import ConfigurationSpace,CSH
from fcvopt.optimizers.fcvopt import FCVOpt
from fcvopt.optimizers.mtbo_cv import MTBOCVOpt
from fcvopt.optimizers.cvrandopt import CVRandOpt
from fcvopt.crossvalidation.sklearn_cvobj import SklearnCVObj
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from argparse import ArgumentParser

parser = ArgumentParser(description='XGboost on Madelon')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--algorithm',type=str,required=True)
parser.add_argument('--n_init',type=int,required=True)
parser.add_argument('--n_iter',type=int,required=True)
parser.add_argument('--n_folds',type=int,default=10)
parser.add_argument('--n_repeats',type=int,default=1)
parser.add_argument('--seed',type=int,default=123)
args = parser.parse_args()

save_dir = os.path.join(args.algorithm+'_runs',args.dataset,'seed_%d'%args.seed)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


#%%
if args.dataset == 'bike_sharing':
    dat = pd.read_csv('../data/bike-sharing.csv')
    y = np.log(1+dat['cnt'].values)
    X_dat = dat[
        ['season','yr','mnth','hr','holiday','weekday','workingday',
        'weathersit','temp','hum','windspeed'
        ]
    ]
    ct = ColumnTransformer([
        ('oe',OneHotEncoder(sparse=False),
        ['yr','mnth','hr','weekday','season','weathersit'],
        )
    ],remainder='passthrough').fit(X_dat)
    X = ct.transform(X_dat)
    del X_dat
elif args.dataset =='superconduct':
    dat = pd.read_csv('../data/superconduct.csv')
    X = dat.values[:,:-1]
    y = dat.values[:,-1]
    del dat


#%%
def metric(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#%%
estimator = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.01,
    colsample_bytree=0.2,
    reg_lambda=1e-6,
)

set_seed(args.seed)
cvobj = SklearnCVObj(
    estimator=estimator,
    X=X,y=y,
    loss_metric=metric,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    task='regression',
    input_preprocessor=None,
    needs_proba=False,
    scale_output=True
)

config = ConfigurationSpace(seed=1234)
n_trees = CSH.UniformIntegerHyperparameter('n_estimators',lower=1,upper=1000,log=True)
max_leaf = CSH.UniformIntegerHyperparameter('max_depth',lower=1,upper=10,log=True)
l2_reg= CSH.UniformFloatHyperparameter('reg_lambda',lower=np.exp(-10),upper=np.exp(10),log=True)
lr = CSH.UniformFloatHyperparameter('learning_rate',lower=1e-6,upper=0.1,log=True)
colsample_tree = CSH.UniformFloatHyperparameter('colsample_bytree',lower=0.1,upper=1)
subsample = CSH.UniformFloatHyperparameter('subsample',lower=0.1,upper=1)
config.add_hyperparameters([n_trees,max_leaf,l2_reg,lr,colsample_tree,subsample])
config.generate_indices()

#%% 
set_seed(args.seed)
if args.algorithm=='fcvopt':
    opt = FCVOpt(
        obj=cvobj.cvloss,
        n_folds=cvobj.cv.get_n_splits(),
        n_repeats=1,
        estimation_method='MAP',
        deterministic=False,
        fold_selection_criterion='variance_reduction',
        fold_initialization='stratified',
        config=config,
        correlation_kernel_class=None,
        kappa=2.,
        verbose=2.,
        save_iter=10,
        save_dir = save_dir
    )
elif args.algorithm =='mtbo':
    opt = MTBOCVOpt(
        obj=cvobj.cvloss,
        n_folds=cvobj.cv.get_n_splits(),
        estimation_method='MAP',
        deterministic=False,
        fold_selection_criterion='single-task-ei',
        config=config,
        correlation_kernel_class=None,
        kappa=2.,
        verbose=2.,
        save_iter=10,
        save_dir = save_dir
    )
else:
    opt = CVRandOpt(
        obj=cvobj.cvloss,
        n_folds=cvobj.cv.get_n_splits(),
        n_repeats=1,
        estimation_method='MAP',
        deterministic=False,
        fold_initialization='stratified',
        config=config,
        correlation_kernel_class=None,
        kappa=2.,
        verbose=2.,
        save_iter=10,
        save_dir = save_dir
    )
out = opt.run(args.n_iter,n_init=args.n_init)
# save to disk
opt.save_to_file(save_dir)