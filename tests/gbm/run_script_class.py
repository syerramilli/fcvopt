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
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml

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

save_dir = os.path.join(args.algorithm+'_mcmc_runs',args.dataset,'seed_%d'%args.seed)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


#%%
X,y = fetch_openml(args.dataset,return_X_y=True,data_home='../data/',as_frame=True)
if args.dataset=='churn':
    ct = ColumnTransformer([
        ('oe',OneHotEncoder(sparse=False),
        ['area_code','number_customer_service_calls'],
        )
    ],remainder='passthrough').fit(X)
    X = ct.transform(X)

#%%
def metric(y_true,y_pred):
    return np.sqrt(1-roc_auc_score(y_true,y_pred))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#%%
estimator = Pipeline([
    ('bal',ADASYN()),
    ('gbm',XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.01,
        gamma=1e-6,
        colsample_bytree=0.2,
        reg_lambda=1e-6,
    ))
])

set_seed(args.seed)
cvobj = SklearnCVObj(
    estimator=estimator,
    X=X,y=y.values,
    loss_metric=metric,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    task='classification',
    input_preprocessor=None,
    needs_proba=True
)

config = ConfigurationSpace(seed=1234)
# gbm hyperparameters
n_trees = CSH.UniformIntegerHyperparameter('gbm__n_estimators',lower=10,upper=2000,log=True)
max_leaf = CSH.UniformIntegerHyperparameter('gbm__max_depth',lower=1,upper=10)
l2_reg= CSH.UniformFloatHyperparameter('gbm__reg_alpha',lower=1e-4,upper=1e+5,log=True)
lr = CSH.UniformFloatHyperparameter('gbm__learning_rate',lower=5e-4,upper=0.5,log=True)
colsample_tree = CSH.UniformFloatHyperparameter('gbm__colsample_bytree',lower=0.1,upper=1)
subsample = CSH.UniformFloatHyperparameter('gbm__subsample',lower=0.1,upper=1)
min_child_weight = CSH.UniformIntegerHyperparameter('gbm__min_child_weight',lower=5,upper=50)
config.add_hyperparameters([n_trees,max_leaf,lr,colsample_tree,
                            subsample,min_child_weight])
# adasyn hyperparameters
#sampling = CSH.UniformFloatHyperparameter('bal__sampling_strategy',lower=0.5,upper=1)
n_neighbors = CSH.UniformIntegerHyperparameter('bal__n_neighbors',lower=5,upper=25)

config.add_hyperparameters([n_neighbors])
config.generate_indices()

#%% 
set_seed(args.seed)
if args.algorithm=='fcvopt':
    opt = FCVOpt(
        obj=cvobj.cvloss,
        n_folds=cvobj.cv.get_n_splits(),
        n_repeats=1,
        estimation_method='MCMC',
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