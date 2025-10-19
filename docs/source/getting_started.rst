Getting Started
===============

This guide will help you get up and running with FCVOpt for efficient hyperparameter optimization.

Installation
------------

From Source

.. code-block:: bash

   git clone https://github.com/syerramilli/fcvopt.git
   cd fcvopt
   pip install .

With Optional Dependencies:

.. code-block:: bash

   pip install .[experiments]  # For reproducing the results from the paper
   pip install .[docs]         # For building documentation

Basic Workflow
--------------

FCVOpt follows a simple three-step workflow:

1. Define a cross-validation objective
2. Create a configuration space
3. Run optimization

Step 1: Create a CV Objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fcvopt.crossvalidation import SklearnCVObj
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import make_classification
   from sklearn.metrics import zero_one_loss # misclassification rate

   # Generate sample data
   X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

   # Create CV objective
   cv_obj = SklearnCVObj(
      estimator=RandomForestClassifier(),
      X=X, y=y,
      loss_metric=zero_one_loss,  # Metric to minimize
      task='binary-classification',
      n_splits=5, # 5-fold cross-validation
      rng_seed=42
   )

Step 2: Define Configuration Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical

   # Define hyperparameter search space
   config = ConfigurationSpace()
   config.add([
      Integer('n_estimators', bounds=(50, 1000)),
      Integer('max_depth', bounds=(1, 12), log=True),
      Float('max_features', bounds=(0.1, 1), log=True),
   ])
   config.generate_indices()

Step 3: Run Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fcvopt.optimizers import FCVOpt

   # Initialize optimizer
   optimizer = FCVOpt(
      obj=cv_obj.cvloss,
      n_folds=cv_obj.cv.get_n_splits(),
      config=config,
      acq_function = 'LCB', # 'KG' gives better results but is slower
      tracking_dir='./hp_opt_runs/', # for mlflow tracking,
      experiment_name='rf_hpt_example'
   )

   # Optimize hyperparameters
   best_conf = optimizer.optimize(n_iter=50)

   print(f"Best configuration: {best_conf}")

Complete Example
----------------

Here's a complete example optimizing XGBoost hyperparameters (with early stopping enabled):

.. code-block:: python

   import numpy as np
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import roc_auc_score
   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import XGBoostCVObjEarlyStopping
   from ConfigSpace import ConfigurationSpace, Integer, Float
   import xgboost as xgb

   # Load data
   X, y = load_breast_cancer(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # define metric
   def auc_loss_metric(y_true, y_pred):
       return 1 - roc_auc_score(y_true, y_pred)

   # Create XGBoost CV objective with early stopping
   cv_obj = XGBoostCVObjEarlyStopping(
       estimator=xgb.XGBClassifier(n_estimators=2000, tree_method='approx'),
       X=X_train, y=y_train,
       loss_metric = auc_loss_metric,
       needs_proba = True, # Set to True since ROC-AUC requires probabilities
       n_splits=10,
       task='binary-classification',
       scoring='roc_auc',
       early_stopping_rounds=10,
       rng_seed = 42
   )

   # Define configuration space
   config = ConfigurationSpace()
   config.add([
      Float('learning_rate',bounds=(1e-5,0.95),log=True),
      Integer('max_depth',bounds=(1,12),log=True),
      Float('reg_alpha',bounds=(1e-8,100),log=True),
      Float('reg_lambda',bounds=(1e-8,100),log=True),
      Float('gamma',bounds=(1e-8,100),log=True),
      Float('subsample',bounds=(0.1,1.)),
      Float('colsample_bytree',bounds=(0.1,1.)),
   ])
   config.generate_indices()

   # Initialize optimizer
   optimizer = FCVOpt(
      obj=cv_obj.cvloss,
      n_folds=cv_obj.cv.get_n_splits(),
      config=config,
      acq_function = 'LCB',
      tracking_dir='./hp_opt_runs/', # for mlflow tracking,
      experiment_name='xgb_hpt_example'
   )

   # Run optimization with a budget of 50 trials
   best_conf = optimizer.optimize(n_trials=50)

   # Train final model with best hyperparameters
   best_model = cv_obj.construct_model(dict(best_conf))
   best_model.fit(X_train, y_train)

   # Evaluate on test set
   test_score = best_model.score(X_test, y_test)
   print(f"Test accuracy: {test_score:.4f}")

Next Steps
----------
* Check the :doc:`optimizers` API reference for advanced options
* Learn about :doc:`mlflow_integration` for experiment tracking