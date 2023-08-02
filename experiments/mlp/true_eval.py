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
from fcvopt.crossvalidation.mlp_cvobj import MLPCVObj
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import QuantileTransformer

from sklearn.datasets import fetch_openml
from functools import partial

parser = ArgumentParser(description='MLP Regression')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--runs_dir',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--n_jobs',type=int,default=1)
parser.add_argument('--n_surrogate_evals',type=int,default=500)
args = parser.parse_args()

save_dir = os.path.join(args.save_dir,args.dataset)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


#%%
n_hidden = 2
config = ConfigurationSpace(seed=1234)
config.add_hyperparameters([
    Integer('hsize%d'%i,bounds=(8,256),log=True) for i in range(n_hidden)
])

# dropouts
config.add_hyperparameters([
    Float('dropout%d'%i,bounds=(0,0.5)) for i in range(n_hidden)
])

# remaining hyperparameters
config.add_hyperparameters([
    Float('lr',bounds=(1e-5,0.1),log=True),
    Integer('batch_size',bounds=(32,2048),log=True),
    Float('weight_decay',bounds=(1e-8,1),log=True),
    Float('callbacks__LRScheduler__factor',bounds=(0.1,0.5))
])
config.generate_indices()

#%%
def metric(y_true,y_pred):
    # rrmse
    return np.sqrt(mean_squared_error(y_true,y_pred))/np.std(y_true,ddof=0)

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
   'house_16H':44139,
   'house_sales':44144,
   'superconduct':44148,
   'pol':44133
}

X,y = fetch_openml(
    data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=False,parser='auto'
)
#%% define estimator and cross-validation objective
set_seed(1)
cvobj = MLPCVObj(
    num_hidden=n_hidden,
    activation='SELU',
    max_epochs=100,
    optimizer='AdamW',
    task='regression',
    X=X.astype(np.float32),y=y.reshape(-1,1).astype(np.float32),
    loss_metric=metric,
    n_splits=10,
    n_repeats=1,
    holdout=False,
    scale_output=True,
    input_preprocessor=QuantileTransformer(output_distribution='normal'),
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

n_iter = args.n_surrogate_evals - 10 + 1

training_path = os.path.join(save_dir,'model_train.pt')
if os.path.exists(training_path):
    # resume progress from last time
    # load saved progress
    training = torch.load(os.path.join(save_dir,'model_train.pt'))
    for k,v in training.items():
        setattr(alc_obj,k,v)

    alc_obj.train_confs = [
        config.get_conf_from_array(x.numpy()) for x in training['train_x']
    ]

    # load stat objects
    stats = joblib.load(os.path.join(save_dir,'stats.pkl'))
    for k,v in stats.items():
        setattr(alc_obj,k,v)

    # run the remaining iterations
    num_iters_completed = len(alc_obj.acq_vec)
    _ = alc_obj.run(n_iter-num_iters_completed)
else:
    _ = alc_obj.run(n_iter=n_iter,n_init=10)

# save to disk
alc_obj.save_to_file(save_dir)