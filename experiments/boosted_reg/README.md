# Tune xgboost for regression models

6 datasets used in the benchmark:

| Dataset           | No. of instances | No. of features |
|-------------------|-----------------:|----------------:|
| wine_quality      |             6497 |              11 |
| cpu_act           |             8192 |              21 |
| Ailerons          |            13750 |              33 |
| pol               |            15000 |              26 |
| elevators         |            16599 |              16 |
| superconductivity |            21263 |              79 |

## Sample commands 

1. Run FCVOPT to tune XGBoost on the wine_quality dataset with sequential LCB as the acquisition function. Limited to 110 total evaulations

    ```{bash}
    python run_xgb_fcvopt.py \
    --dataset wine_quality \
    --save_dir runs \
    --acq lcb \
    --n_init 10 \
    --n_iter 111 \
    --seed 234
    ```

2. Run FCVOPT to tune XGBoost on the elevators dataset with batch Knowledge Gradient (4 candidates per batch) as the acquisition function. Limited to 110 total evaulations

    ```{bash}
    python run_xgb_fcvopt.py \
    --dataset elevators \
    --save_dir runs \
    --acq kg_batch \
    --n_init 10 \
    --n_iter 26 \
    --seed 234
    ```