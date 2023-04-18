import numpy as np
import pandas as pd
import os,joblib
import torch
import random

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer

from fcvopt.optimizers.fcvopt import FCVOpt
from fcvopt.crossvalidation.sklearn_cvobj import SklearnCVObj
from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.datasets import fetch_openml


from argparse import ArgumentParser

parser = ArgumentParser(description='PCA-SVM classification')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument(
    '--acq',
    type=str,required=True,
    choices=['lcb','lcb_batch','kg_batch']
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
    'gina_agnostic':1038,
    'hiva_agnostic':1039,
    'madelon':1485,
    'bioresponse':4134,
}


X,y = fetch_openml(data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=False)

#%% define estimator and cross-validation objective
set_seed(args.seed)
estimator = Pipeline([
    ('scaler',StandardScaler()),
    ('pca',PCA()),
    ('svm',SVC(probability=True))
])

cvobj = SklearnCVObj(
    estimator=estimator,
    X=X,y=y,
    loss_metric=metric,
    needs_proba=True,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    task='binary-classification',
)


#%% parse acquistion arguments
acq_args = {}
acq_args['acq_function'] = 'LCB' if 'lcb' in args.acq else 'KG'
if 'batch' in args.acq:
    acq_args['batch_acquisition']=True
    acq_args['acquisition_q']=4

#%% 
set_seed(args.seed)
config = ConfigurationSpace(seed=1234)
config.add_hyperparameters([
    Float('svm__C',(1e-5,1e+5),log=True),
    Float('svm__gamma',(1e-5,1e+5),log=True),
    Integer('pca__n_components',(1,200),log=True)
])
config.generate_indices()

#%%
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