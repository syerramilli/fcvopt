MLflow Integration
==================

FCVOpt integrates with `MLflow <https://mlflow.org/>`_ for automatic experiment tracking during hyperparameter optimization.

What Gets Tracked
-----------------

FCVOpt automatically logs:

* Hyperparameter configurations evaluated
* Cross-validation scores and best values found
* Number of fold evaluations and optimization progress
* Acquisition function settings and fold selection strategy
* Optimization runtime and convergence metrics

Tracking Options
----------------

Local Directory Tracking:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt

   # Track to local directory
   optimizer = FCVOpt(
       obj=cv_obj.cvloss,
       config=config_space,
       tracking_dir='./my_experiments/',  # Local directory
       experiment_name='rf_optimization'
   )

Remote MLflow Server:

.. code-block:: python

   # Track to remote MLflow server
   optimizer = FCVOpt(
       obj=cv_obj.cvloss,
       config=config,
       tracking_uri='http://localhost:5000',  # MLflow server URL
       experiment_name='rf_optimization'
   )

Continuing Runs
---------------

Continue optimization in the current session:

.. code-block:: python

   # Initial run
   optimizer = FCVOpt(obj=cv_obj.cvloss, config=config_space, tracking_dir='./experiments/')
   best_conf = optimizer.optimize(n_trials=50)

   # Continue with more trials in same session
   best_conf = optimizer.optimize(n_trials=25)  # Additional 25 trials

Restoring Runs
--------------

Restore and continue optimization in a new session:

.. code-block:: python

   # define your CV objective as before
   cv_obj = ...

   # In a new session, restore previous run
   restored_optimizer = FCVOpt.restore_from_mlflow(
       obj=cv_obj.cvloss,
       run_id='your_run_id_here',
       n_folds=5,  # Must match original optimization
       tracking_dir='./experiments/'
   )

   # Continue optimization with additional trials
   best_config = restored_optimizer.optimize(n_trials=50)

MLflow UI
---------

View your experiments in the MLflow web interface:

.. code-block:: bash

   # Navigate to your tracking directory
   cd <path_to_your_tracking_dir>

   # Launch MLflow UI
   mlflow ui

   # Open http://localhost:5000 in your browser

The UI allows you to:

* Track experiment metadata and performance metrics
* Export results