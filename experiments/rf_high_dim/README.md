# Tuning random forest classifier for large dimensional datasets

## Bash scripts to perform a single run for each algorithm

Notes:
1. Setting the seed fixes the initial set of configurations for all algorithms. 
2. All the scripts will create a new directory inside the path specified by the `save_dir` argument. 
    - For FCVOPT (and MTBO), the directory will be the same as the `acq` argument.
    - For SMAC and Optuna, the directories are 'SMAC' and 'optuna' respectively.
3. Each of the scripts accepts 4 dataset arguments: 'gina', 'hiva' (not included in the paper), 'madelon', 'bioresponse'.

### FCVOPT

```
python run_fcvopt.py \
--dataset <dataset_name> \
--save_dir <path_to_outputs> \
--acq kg \
--n_init 10 \
--n_iter 91 \
--seed 100
```

### MTBO
```
python run_fcvopt.py \
--dataset <dataset_name> \
--save_dir <path_to_outputs> \
--acq mtbo \
--n_init 10 \
--n_iter 91 \
--seed 100
```

### SMAC
```
python run_smac.py \
--dataset <dataset_name> \
--save_dir <path_to_outputs> \
--n_init 10 \
--n_iter 91 \
--seed 100
```

### Optuna
```
python run_optuna.py \
--dataset <dataset_name> \
--save_dir <path_to_outputs> \
--n_init 10 \
--n_iter 91 \
--seed 100
```

## Bash script to train a surrogate model for the 'true' CV loss

After completing the experiments for ALL optimizers for a dataset, run `true_eval.py` with the following syntax

```
python true_eval.py \
--dataset <dataset_name> \
--runs_dir <path_to_opt_runs> \
--save_dir <path_to_save_true_model> \
--n_jobs <number_of_cores_available>
```

### Notes:

1. This script is expensive to run.
2. Data is added to the model sequentially using an active learning framework. 
3. If there are multiple cores available, the parallelization is performed over the fold evaluations. 