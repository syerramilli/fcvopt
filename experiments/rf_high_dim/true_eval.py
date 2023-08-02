import os
import torch
import numpy as np
import pandas as pd
import random 

import joblib

from smac.runhistory import RunHistory
from smac import Scenario
from smac.intensifier import Intensifier
from pathlib import Path


from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Configuration
from ConfigSpace import Float,Integer
from fcvopt.optimizers.active_learning import ActiveLearning

from argparse import ArgumentParser
from fcvopt.crossvalidation.sklearn_cvobj import SklearnCVObj
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_openml
from functools import partial

parser = ArgumentParser(description='Random forest classification')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--runs_dir',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--n_jobs',type=int,default=1)
parser.add_argument('--n_surrogate_evals',type=int,default=250)
args = parser.parse_args()

save_dir = os.path.join(args.save_dir,args.dataset)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#%%
config = ConfigurationSpace(seed=1234)
config.add_hyperparameters([
    Integer('max_depth',bounds=(1,12),log=True),
    Float('min_impurity_decrease',bounds=(1e-8,10),log=True),
    Float('max_features',bounds=(0.005,0.5),log=True),
    Integer('min_samples_split',bounds=(5,250),log=True),
])
config.generate_indices()

#%%
def metric(y_true,y_pred):
    return np.sqrt(1-roc_auc_score(y_true,y_pred))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_confs(save_dir,config=None,algorithm='inhouse'):
    folders = sorted(os.listdir(save_dir))
    out = []
    for folder in folders:
        try:
            if algorithm == 'inhouse':
                tmp = joblib.load(os.path.join(
                    save_dir,folder,'stats.pkl'
                ))['confs_inc']
            elif algorithm == 'optuna':
                study = joblib.load(os.path.join(
                    save_dir,folder,'study.pkl'
                ))
                
                best_loss = np.inf
                tmp = []
                for trial in study.trials:
                    if best_loss < trial.values[0]:
                        tmp.append(tmp[-1])
                    else:
                        tmp.append(Configuration(config,values=trial.params))
                        best_loss = trial.values[0]
                        
                tmp = tmp[9:]
            elif algorithm == 'SMAC':
                full_path = os.path.join(save_dir,folder)
                subfolder = os.listdir(full_path)[0]
                
                final_path = Path(full_path,subfolder,'0')
                
                scenario = Scenario.load(final_path)
                
                runhistory = RunHistory()
                runhistory.load(final_path/'runhistory.json',configspace=scenario.configspace)
                
                id_configs = {v:k for k,v in runhistory.config_ids.items()}
                
                intensifier=Intensifier(scenario)
                intensifier._runhistory = runhistory
                intensifier.load(final_path/'intensifier.json')
                
                trial_history = []
                trial_updates = []
                for t in intensifier.trajectory:
                    trial_history.append(id_configs[t.config_ids[0]])
                    trial_updates.append(t.trial)

                reps = np.diff(np.concatenate([trial_updates,[scenario.n_trials+1]]))
                    
                tmp = []
                for i,conf in enumerate(trial_history):
                    tmp.extend([conf]*reps[i])

                tmp = tmp[9:]
                
            out.append(tmp)
        except Exception as e:
            print(e)
            print(folder)
    return out

#%% fetch dataset
DATA_IDS = {
    'gina':1038,
    'hiva':1039,
    'madelon':1485,
    'bioresponse':4134,
}


X,y = fetch_openml(data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=False)

#%% define estimator and cross-validation objective
set_seed(1)
cvobj = SklearnCVObj(
    estimator=RandomForestClassifier(n_estimators=500),
    X=X,y=y,
    loss_metric=metric,
    needs_proba=True,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    task='binary-classification',
    num_jobs=args.n_jobs
)

#%%
acqfuncs = ['kg','kg_batch','optuna','SMAC']

lists_conf_lists = []
for acqfunc in acqfuncs:
    algorithm = acqfunc if acqfunc in ['optuna','SMAC'] else 'inhouse'
    #algorithm = 'optuna' if acqfunc == 'optuna' else 'inhouse'
    lists_conf_lists.append(
        load_confs(args.runs_dir + '%s/%s/'%(args.dataset,acqfunc),config,algorithm)
    )

confs_unq = []

unique_confs = set()

for list_confs_list in lists_conf_lists:
    for confs_list in list_confs_list:
        for conf in confs_list:
            if conf not in unique_confs:
                unique_confs.add(conf)
            
unique_confs = list(unique_confs)

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
_ = alc_obj.run(n_iter=args.n_surrogate_evals-10+1,n_init=10) # args.n_surrogate_evals evaluations

# save to disk
alc_obj.save_to_file(save_dir)