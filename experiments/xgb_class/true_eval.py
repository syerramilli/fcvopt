import os
import torch
import numpy as np
import pandas as pd
import random 

import joblib

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer,Categorical
from fcvopt.optimizers.active_learning import ActiveLearning

from argparse import ArgumentParser
from fcvopt.crossvalidation.sklearn_cvobj import XGBoostCVObjEarlyStopping
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from sklearn.datasets import fetch_openml
from functools import partial

parser = ArgumentParser(description='XGBoost classification')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--n_jobs',type=int,required=True)
args = parser.parse_args()

RUNS_DIR = 'runs/'

save_dir = os.path.join(args.save_dir,args.dataset)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


#%%
def metric(y_true,y_pred):
    return np.sqrt(1-roc_auc_score(y_true,y_pred))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_confs(save_dir):
    folders = sorted(os.listdir(save_dir))
    out = []
    for folder in folders:
        try:
            tmp = joblib.load(os.path.join(
                save_dir,folder,'stats.pkl'
            ))['confs_inc']
            out.append(tmp)
        except:
            print(folder)
    return out

#%% fetch dataset
DATA_IDS = {
    'compass':44162,
    'eye_movements':44157,
    'electricity':44156
}

X,y = fetch_openml(
    data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=True,parser='auto'
)
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
    early_stopping_rounds=50,
    num_jobs=args.n_jobs
)

#%%
lists_conf_lists = [
    load_confs(RUNS_DIR + '%s/%s/'%(args.dataset,s)) for s in ['kg','kg_batch']]
confs_unq = []

unique_confs = set()

for list_confs_list in lists_conf_lists:
    for confs_list in list_confs_list:
        for conf in confs_list:
            if conf not in unique_confs:
                unique_confs.add(conf)
            
unique_confs = list(unique_confs)

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
set_seed(5)
X_ref = torch.from_numpy(
    np.row_stack([
        conf.get_array() for conf in unique_confs
    ])
)

alc_obj = ActiveLearning(
    obj = cvobj.cvloss,
    config=config,
    X_ref=X_ref,
    n_jobs=1,
    verbose=2,
    save_dir=save_dir,
    save_iter=1
)
_ = alc_obj.run(n_iter=241,n_init=10) # 200 evaluations

# save to disk
alc_obj.save_to_file(save_dir)