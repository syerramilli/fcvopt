FCVOpt
======

FCVOpt is a Python package for hyperparameter optimization via Fractional Cross-Validation. It implements the methodology from `"Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms" <https://doi.org/10.1080/00401706.2025.2515926>`_ using hierarchical Gaussian processes to efficiently optimize ML models by evaluating only a fraction of CV folds.

K-fold cross-validation is more robust than holdout validation, but requires fitting K models per hyperparameter configuration—making it expensive inside an optimization loop. FCVOpt sidesteps this by modeling the correlation structure of fold-wise losses across the hyperparameter space with a hierarchical GP, so that most configurations need only a single fold evaluated.

Features
--------

* Fractional CV optimization via hierarchical Gaussian processes, with support for repeated K-fold cross-validation
* Standard Bayesian optimization with holdout loss, available for both hyperparameter tuning and general black-box optimization
* Fold selection via variance reduction, which chooses the most informative fold to evaluate at each step
* MLflow integration for experiment tracking and model checkpointing
* Acquisition functions: Knowledge Gradient and Lower Confidence Bound
* Works with scikit-learn estimators, XGBoost, and neural networks (via PyTorch-Skorch)

Installation
------------

.. code-block:: bash

   git clone https://github.com/syerramilli/fcvopt.git
   cd fcvopt
   pip install .

Quick Start
-----------

.. code-block:: python

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

Research
--------

FCVOpt implements the algorithm described in:

| "Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms"
| *Suraj Yerramilli and Daniel W. Apley*
| *Technometrics* (2025)
| DOI: `10.1080/00401706.2025.2515926 <https://doi.org/10.1080/00401706.2025.2515926>`_

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   examples/01_Introduction_to_FCVOpt.ipynb
   examples/02_Tuning_Lightgbm_Sklearn_API.ipynb
   examples/03_Extending_CVobjective.ipynb
   examples/04_Standard_BO.ipynb

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   optimizers
   crossvalidation
   models
   configspace

.. toctree::
   :maxdepth: 1
   :caption: Advanced:
   
   technical_details
   mlflow_integration

Indices and Tables
======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
