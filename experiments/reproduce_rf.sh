#!/bin/bash
cd rf_high_dim

DATASET=madelon
RUNS_DIR=opt_runs # this will be inside rf_high_dim
FIG_DIR=figures_auto # this will be in the main experiments folders
N_INIT=10
N_ITER=91

# run all the experiments
for SEED IN 1000 1107 1214 1321 1428 1535 1642 1750 1857 1964 2071 2178 2285 2392 2500
do
    python run_fcvopt.py \
    --dataset $DATASET \
    --save_dir $RUNS_DIR \
    --acq kg \
    --n_init $N_INIT\
    --n_iter $N_ITER \
    --seed $SEED

    python run_fcvopt.py \
    --dataset $DATASET \
    --save_dir $RUNS_DIR \
    --acq mtbo \
    --n_init $N_INIT\
    --n_iter $N_ITER \
    --seed $SEED

    python run_smac.py \
    --dataset $DATASET \
    --save_dir $RUNS_DIR \
    --n_init $N_INIT\
    --n_iter $N_ITER \
    --seed $SEED

    python run_optuna.py \
    --dataset $DATASET \
    --save_dir $RUNS_DIR \
    --n_init $N_INIT\
    --n_iter $N_ITER \
    --seed $SEED
done

# generate the plots - store in directory $FIG_DIR
# if passing the --use_tex flag, a latex distribution
# must be installed and found in the PATH
cd ../
LOAD_DIR="rf_high_dim/$RUNS_DIR"

python generate_figures.py \
--runs_dir $LOAD_DIR \
--true_cv_models_dir rf_high_dim/true_cv_models \
--model rf \
--dataset $DATASET \
--save_dir $FIG_DIR #--use_tex 