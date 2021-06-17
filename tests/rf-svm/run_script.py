import numpy as np
import pandas as pd
import os,joblib
import torch
import random

from fcvopt.configspace import ConfigurationSpace,CSH
from fcvopt.optimizers.fcvopt import FCVOpt
from fcvopt.crossvalidation.sklearn_cvobj import SklearnCVObj
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

parser = ArgumentParser(description='PCA-SVM on Madelon')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--n_init',type=int,required=True)
parser.add_argument('--n_iter',type=int,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--n_folds',type=int,default=10)
parser.add_argument('--n_repeats',type=int,default=1)
parser.add_argument('--seed',type=int,default=123)
args = parser.parse_args()

save_dir = os.path.join(args.save_dir,args.dataset,'seed_%d'%args.seed)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


#%%
if args.dataset == 'madelon':
    dat = pd.read_csv('../data/madelon.csv')
    X= dat.values[:,:500]
    y = dat.values[:,500]
elif args.dataset == 'gina':
    dat = pd.read_csv('../data/gina.csv')
    X = dat.values[:,1:]
    y = dat.values[:,0]
    del dat
elif args.dataset == 'arcene':
    dat = pd.read_csv('../data/arcene.csv')
    X = dat.values[:,:-1]
    y = dat.values[:,-1]
    del dat

#%%
def metric(y_true,y_pred):
    return np.sqrt(1-roc_auc_score(y_true,y_pred))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#%%
estimator = Pipeline([
    ('fs',SelectFromModel(
        RandomForestClassifier(min_samples_leaf=20),
        threshold=-np.inf
    )),
    ('svm',SVC(gamma=0.1,probability=True))
])

set_seed(args.seed)
svmcv = SklearnCVObj(
    estimator=estimator,
    X=X,y=y,
    loss_metric=metric,
    needs_proba=True,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    task='binary-classification',
    input_preprocessor=StandardScaler()
)

config = ConfigurationSpace(seed=1234)
C= CSH.UniformFloatHyperparameter('svm__C',lower=1e-5,upper=1e+5,log=True)
gamma= CSH.UniformFloatHyperparameter('svm__gamma',lower=1e-5,upper=1e+5,log=True)
# random forest hyperparameters
rf_num_trees = CSH.UniformIntegerHyperparameter(
    'fs__estimator__n_estimators',lower=1,upper=1000,log=True
)
rf_max_feat = CSH.UniformIntegerHyperparameter(
    'fs__estimator__max_features',lower=1,upper=X.shape[1],log=True
)
# feature selection hyperparameters
fs_max_feat = CSH.UniformIntegerHyperparameter(
    'fs__max_features',lower=1,upper=0.5*X.shape[1],log=True
)


config.add_hyperparameters([C,gamma,rf_num_trees,rf_max_feat,fs_max_feat])
config.generate_indices()
#%% 
set_seed(args.seed)
opt = FCVOpt(
    obj=svmcv.cvloss,
    n_folds=svmcv.cv.get_n_splits(),
    n_repeats=1,
    estimation_method='MAP',
    deterministic=False,
    fold_selection_criterion='variance_reduction',
    fold_initialization='stratified',
    config=config,
    correlation_kernel_class=None,
    kappa=2.,
    verbose=2.
)
out = opt.run(args.n_iter,n_init=args.n_init)
# save to disk
opt.save_to_file(save_dir)