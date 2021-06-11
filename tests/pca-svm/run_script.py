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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

parser = ArgumentParser(description='PCA-SVM on Madelon')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--n_init',type=str,required=True)
parser.add_argument('--n_iter',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--n_folds',type=int,default=10)
parser.add_argument('--n_repeats',type=int,default=1)
parser.add_argument('--seed',type=int,default=123)
args = parser.parse_args()

save_dir = os.path.join(args.save_dir,'seed_$d'%args.seed)
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
    ('pca',PCA(n_components=1)),
    ('svm',SVC(gamma=0.1,probability=True))
])

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
pca_comp = CSH.UniformIntegerHyperparameter('pca__n_components',lower=1,upper=X.shape[1]*0.5,log=True)
C= CSH.UniformFloatHyperparameter('svm__C',lower=1e-5,upper=1e+5,log=True)
gamma= CSH.UniformFloatHyperparameter('svm__gamma',lower=1e-5,upper=1e+5,log=True)
config.add_hyperparameters([pca_comp,C,gamma])
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
    verbose=0.
)
out = opt.run(args.n_iter,n_init=args.n_init)
# save to disk
opt.save_to_file(save_dir)