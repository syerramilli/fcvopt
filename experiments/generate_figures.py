import os
import torch
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ConfigSpace import Configuration, Float, Integer, Categorical
from fcvopt.configspace import ConfigurationSpace

from smac.runhistory import RunHistory
from smac import Scenario
from smac.intensifier import Intensifier
from pathlib import Path
import logging
logging.disable(logging.CRITICAL)

# Import MLflow for loading new format data
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    import tempfile
    import json
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Only old format (stats.pkl) will be supported.")

#%%
parser = argparse.ArgumentParser(description='Generate figures from results')
parser.add_argument('--runs_dir', type=str, required=True)
parser.add_argument('--true_cv_models_dir', type=str, required=True)
parser.add_argument('--model', type=str, required=True, choices=['rf', 'mlp', 'xgb', 'tab_resnet'])
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--use_tex', action='store_true')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

#%% configure plots
plt.rcParams.update(
    {'figure.dpi':150,
     'font.family':'serif',
     'text.usetex':args.use_tex
    }
)
plt.style.use('ggplot')
sns.set_palette('tab10')

#%%

config = ConfigurationSpace(seed=1234)
if args.model == 'rf':
    config.add([
        Integer('max_depth',bounds=(1,12),log=True),
        Float('min_impurity_decrease',bounds=(1e-8,10),log=True),
        Float('max_features',bounds=(0.005,0.5),log=True),
        Integer('min_samples_split',bounds=(5,250),log=True),
    ])

    loss_fn_string = r'$\sqrt{1-\mathrm{AUC}}$'
    model_string = 'RF'
elif args.model == 'mlp':
    n_hidden = 2
    config.add([
        Integer('hsize%d'%i,bounds=(8,256),log=True) for i in range(n_hidden)
    ])

    # dropouts
    config.add([
        Float('dropout%d'%i,bounds=(0,0.5)) for i in range(n_hidden)
    ])

    # remaining hyperparameters
    config.add([
        Float('lr',bounds=(1e-5,0.1),log=True),
        Integer('batch_size',bounds=(32,2048),log=True),
        Float('weight_decay',bounds=(1e-8,1),log=True),
        Float('callbacks__LRScheduler__factor',bounds=(0.1,0.5))
    ])

    loss_fn_string = 'RRMSE'
    model_string = 'MLP'

elif args.model == 'xgb':
    config.add([
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
    loss_fn_string = r'$\sqrt{1-\mathrm{AUC}}$'
    model_string = 'XGboost'

elif args.model == 'tab_resnet':
    config.add([
        Integer('n_hidden', bounds=(1, 6), log=True),
        Integer('layer_size', bounds=(8, 512), log=True),
        Categorical('normalization', ['batchnorm', 'layernorm']),
        Float('hidden_factor', bounds=(1, 4)),
        Float('hidden_dropout', bounds=(0, 0.5)),
        Float('residual_dropout', bounds=(0, 0.5)),
        Float('lr', bounds=(1e-5, 0.1), log=True),
        Float('weight_decay', bounds=(1e-8, 1e-2), log=True),
    ])
    loss_fn_string = 'RRMSE'
    model_string = 'Tabular ResNet'    

config.generate_indices()

#%% Function for loading results

def load_mlflow_confs(seed_dir):
    """Load incumbent configurations from MLflow artifacts."""
    if not MLFLOW_AVAILABLE:
        return None

    try:
        # Find MLflow artifacts by scanning for directory containing artifacts
        for root, dirs, files in os.walk(seed_dir):
            if 'artifacts' in dirs:
                artifacts_dir = os.path.join(root, 'artifacts')
                iterations_dir = os.path.join(artifacts_dir, 'iterations')

                if os.path.exists(iterations_dir):
                    # Load iteration files and extract incumbent configurations
                    iteration_files = [f for f in os.listdir(iterations_dir) if f.startswith('iter_') and f.endswith('.json')]
                    iteration_files.sort()  # Ensure chronological order

                    confs_inc = []
                    for iter_file in iteration_files:
                        iter_path = os.path.join(iterations_dir, iter_file)
                        with open(iter_path, 'r') as f:
                            iter_data = json.load(f)

                        # Extract incumbent configuration from iteration
                        if 'conf_inc' in iter_data:
                            conf_dict = iter_data['conf_inc']
                            conf = Configuration(config, values=conf_dict)
                            confs_inc.append(conf)

                    return confs_inc if confs_inc else None

        return None

    except Exception as e:
        print(f"Error loading MLflow data from {seed_dir}: {e}")
        return None

def load_confs(save_dir,algorithm='inhouse'):
    folders = sorted(os.listdir(save_dir))
    out = []
    for folder in folders:
        try:
            folder_path = os.path.join(save_dir, folder)

            if algorithm == 'inhouse':
                # First try to load from MLflow format
                mlflow_confs = load_mlflow_confs(folder_path)
                if mlflow_confs is not None:
                    tmp = mlflow_confs
                else:
                    # Fall back to old stats.pkl format
                    tmp = joblib.load(os.path.join(
                        folder_path,'stats.pkl'
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

                intensifier = Intensifier(scenario)
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

#%% Load results
true_model = torch.jit.load(os.path.join(args.true_cv_models_dir, f'{args.dataset}.pt'))
acqfuncs = ['kg', 'mtbo', 'SMAC', 'optuna']

runs_dir_dataset = os.path.join(args.runs_dir, args.dataset)
res_true_dict = {}

for i in range(len(acqfuncs)):
    tmp_mean = []
    tmp_std = []
    
    algorithm = 'inhouse'
    if acqfuncs[i] == 'optuna':
        algorithm = 'optuna'
    elif acqfuncs[i] == 'SMAC':
        algorithm = 'SMAC'
        
    lists_conf_list = load_confs(
        os.path.join(runs_dir_dataset,acqfuncs[i]),
        algorithm=algorithm
    )
    for confs_list in lists_conf_list:
        test_x = torch.from_numpy(
            np.array([conf.get_array() for conf in confs_list])
        ).double()

        with torch.no_grad():
            pred, _ = true_model(test_x)

        tmp_mean.append(pred.numpy())

    res_true_dict[acqfuncs[i]] = np.column_stack(tmp_mean)

#%% Plots
labels_dict = {
    'kg': 'FCVOPT',
    'mtbo': 'MTBO-CV',
    'SMAC':'SMAC',
    'optuna':'Optuna'
}

def progress_plot_mean(x, arr_list , ax, label):
    mean = np.mean(arr_list,axis=1)
    std_dev = np.std(arr_list,axis=1)/np.sqrt(arr_list.shape[1])
    ax.plot(x,mean,label=label)
    ax.fill_between(
        x,
        mean-std_dev,mean+std_dev,
        alpha=0.5,
    )
    return ax

fig,ax = plt.subplots(1,1,figsize=(6,4))
for acqfunc,res_true in res_true_dict.items():
    
    _ = progress_plot_mean(
        10 + np.arange(res_true.shape[0]),
        res_true,
        ax,
        label=labels_dict[acqfunc]
    )
    

_ = ax.legend()
_ = ax.set_xlabel('Number ($N$) of fold evaluations')
_ = ax.set_ylabel(r'$f_\mathrm{true}\left(\mathbf{x}_\mathrm{inc}^{(N)}\right)$ for ' + loss_fn_string)
_ = ax.set_title(f'{model_string}: {args.dataset}')
    
fig.savefig(os.path.join(args.save_dir, f'fig-{args.model}-{args.dataset}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(args.save_dir, f'fig-{args.model}-{args.dataset}.png'), bbox_inches='tight')