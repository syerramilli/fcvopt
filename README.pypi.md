# fcvopt: Fractional cross-validation for hyperparameter optimization

FCVOpt is a Python package for hyperparameter optimization via Fractional Cross-Validation. It implements the methodology from ["Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms"](https://doi.org/10.1080/00401706.2025.2515926) using hierarchical Gaussian processes to efficiently optimize ML models by evaluating only a fraction of CV folds.

K-fold cross-validation is more robust than holdout validation, but requires fitting K models per hyperparameter configuration—making it expensive inside an optimization loop. FCVOpt sidesteps this by modeling the correlation structure of fold-wise losses across the hyperparameter space with a hierarchical GP, so that most configurations need only a single fold evaluated.

The documentation is available at [https://syerramilli.github.io/fcvopt/](https://syerramilli.github.io/fcvopt/).

## Features

* Fractional CV optimization via hierarchical Gaussian processes, with support for repeated K-fold cross-validation
* Standard Bayesian optimization with holdout loss, available for both hyperparameter tuning and general black-box optimization
* Fold selection via variance reduction, which chooses the most informative fold to evaluate at each step
* MLflow integration for experiment tracking and model checkpointing
* Acquisition functions: Knowledge Gradient and Lower Confidence Bound
* Works with scikit-learn estimators, XGBoost, and neural networks (via PyTorch-Skorch)

## Installation

```bash
pip install fcvopt
```

## Quick Start

```python
from fcvopt.optimizers import FCVOpt
from fcvopt.crossvalidation import SklearnCVObj
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss
from fcvopt.configspace import ConfigurationSpace
from ConfigSpace import Integer, Float


# Define the CV objective
cv_obj = SklearnCVObj(
   estimator=RandomForestClassifier(),
   X=X, y=y,
   loss_metric=zero_one_loss,
   task='binary-classification',
   n_splits=5,
   rng_seed=42
)

# Define the hyperparameter search space
config = ConfigurationSpace()
config.add([
   Integer('n_estimators', bounds=(10, 1000), log=True),
   Integer('max_depth', bounds=(1, 12), log=True),
   Float('max_features', bounds=(0.1, 1), log=True),
])
config.generate_indices()

# Set up the optimizer
optimizer = FCVOpt(
   obj=cv_obj.cvloss,
   n_folds=cv_obj.cv.get_n_splits(),
   config=config,
   acq_function='LCB',           # 'KG' tends to work better but is slower
   fold_selection_criterion='variance_reduction',
   tracking_dir='./hpt_opt_runs/',
   experiment_name='rf_hpt'
)

# Run 50 trials, using 10 random initializations before switching to acquisition
best_conf = optimizer.optimize(n_trials=50, n_init=10)
optimizer.end_run()
```

## Research

FCVOpt implements the algorithm described in:

> "Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms"
> Suraj Yerramilli and Daniel W. Apley
> *Technometrics* (2025)
> DOI: [10.1080/00401706.2025.2515926](https://doi.org/10.1080/00401706.2025.2515926)

## Citing

If you use this code in your research, please cite the following paper:

```
@article{yerramilli2025fractional,
    author = {Suraj Yerramilli and Daniel W. Apley},
    title = {Fractional Cross-Validation for Optimizing Hyperparameters of Supervised Learning Algorithms},
    journal = {Technometrics},
    year = {2025},
    doi = {10.1080/00401706.2025.2515926},
}
```
