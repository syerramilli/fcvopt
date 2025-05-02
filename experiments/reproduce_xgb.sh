#!/bin/bash
cd xgb_class

DATASET=eye_movements
RUNS_DIR=runs # this will be inside xgb_class
FIG_DIR=figures_auto # this will be in the main experiments folders
N_INIT=10
N_ITER=101

# run all the experiments
for SEED in 1000 1107 1214 1321 1428 1535 1642 1750 1857 1964 2071 2178 2285 2392 2500
do
    python3 run_xgb_fcvopt.py \
    --dataset $DATASET \
    --save_dir $RUNS_DIR \
    --acq kg \
    --n_init $N_INIT\
    --n_iter $N_ITER \
    --seed $SEED

    python3 run_xgb_fcvopt.py \
    --dataset $DATASET \
    --save_dir $RUNS_DIR \
    --acq mtbo \
    --n_init $N_INIT\
    --n_iter $N_ITER \
    --seed $SEED

    python3 run_xgb_smac.py \
    --dataset $DATASET \
    --save_dir $RUNS_DIR \
    --n_init $N_INIT\
    --n_iter $N_ITER \
    --seed $SEED

    python3 run_xgb_optuna.py \
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
LOAD_DIR="xgb_class/$RUNS_DIR"

python3 generate_figures.py \
--runs_dir $LOAD_DIR \
--true_cv_models_dir xgb_class/true_cv_models \
--model xgb \
--dataset $DATASET \
--save_dir $FIG_DIR #--use_tex 