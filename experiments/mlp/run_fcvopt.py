import numpy as np
import pandas as pd
import os,joblib
import torch
import random

from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Float,Integer

from fcvopt.optimizers.fcvopt import FCVOpt
from fcvopt.optimizers.mtbo_cv import MTBOCVOpt
from fcvopt.crossvalidation.mlp_cvobj import MLPCVObj
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import QuantileTransformer

from sklearn.datasets import fetch_openml

from argparse import ArgumentParser

parser = ArgumentParser(description='MLP regression')
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
    # rrmse
    return np.sqrt(mean_squared_error(y_true,y_pred))/np.std(y_true,ddof=0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#%% fetch dataset
DATA_IDS = {
   'house_16H':44139,
   'house_sales':44144,
   'superconduct':44148,
   'pol':44133
}

X,y = fetch_openml(data_id=DATA_IDS[args.dataset],return_X_y=True,as_frame=False,parser='auto')

#%% define estimator and cross-validation objective
set_seed(1)
n_hidden = 2
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
    input_preprocessor=QuantileTransformer(output_distribution='normal')
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
set_seed(args.seed)
if args.acq == 'mtbo':
    opt = MTBOCVOpt(
        obj=cvobj.cvloss,
        n_folds=cvobj.cv.get_n_splits(),
        fold_initialization='stratified',
        config=config,
        verbose=2,
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
        verbose=2,
        **acq_args
    )

out = opt.run(args.n_iter,n_init=args.n_init)