Welcome to FCVOpt's Documentation!
====================================

FCVOpt is a Python package for Fractional Cross-Validation in hyperparameter optimization. It implements the methodology from `"Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms" <https://doi.org/10.1080/00401706.2025.2515926>`_ using hierarchical Gaussian processes to efficiently optimize ML models by evaluating only a fraction of CV folds.

Key Innovation: While K-fold cross-validation is more robust than holdout validation, it is computationally expensive since models must be fit K times at each hyperparameter configuration. FCVOpt addresses this by exploiting the correlation structure between folds across the hyperparameter space, requiring evaluation of only a single fold for many configurations.

🚀 Key Features
-------------------

* Efficient Optimization: Evaluate hyperparameters using only a subset of CV folds via hierarchical Gaussian processes
* Standard Bayesian Optimization: Available for hyperparameter optimization with holdout loss and for general purpose optimization
* Intelligent Fold Selection: Variance reduction strategy that selects which CV folds to evaluate at each configuration
* MLflow Integration: Automatic experiment tracking and model versioning
* Multiple Acquisition Functions: Knowledge Gradient, Lower Confidence Bound
* Framework Support: Scikit-learn, XGBoost, Neural Networks, and more

📖 Quick Start
------------------

Install FCVOpt:

.. code-block:: bash

   git clone https://github.com/syerramilli/fcvopt.git
   cd fcvopt
   pip install .

Basic usage:

.. code-block:: python

   from fcvopt.optimizers import FCVOpt
   from fcvopt.crossvalidation import SklearnCVObj
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import zero_one_loss
   from fcvopt.configspace import ConfigurationSpace
   from ConfigSpace import Integer, Float


   # Create CV objective
   cv_obj = SklearnCVObj(
      estimator=RandomForestClassifier(),
      X=X, y=y,
      loss_metric=zero_one_loss,  # Metric to minimize
      task='binary-classification',
      n_splits=5, # 5-fold cross-validation
      rng_seed=42
   )

   # define the hyperparameter configuration space
   config = ConfigurationSpace()
   config.add([
      Integer('n_estimators', bounds=(10,1000), log=True),
      Integer('max_depth', bounds=(1,12), log=True),
      Float('max_features', bounds=(0.1, 1), log=True),
   ])
   config.generate_indices()

   # Initialize optimizer
   optimizer = FCVOpt(
      obj=cv_obj.cvloss,
      n_folds=cv_obj.cv.get_n_splits(),
      config=config,
      acq_function = 'LCB', # 'KG' gives better results but is slower
      tracking_dir='./hp_opt_runs/', # for mlflow tracking,
      experiment_name='rf_hpt'
   )
   
   # run optimization with a budget of 50 trials
   best_conf = optimizer.optimize(n_trials=50)

🔬 Research Background
--------------------------

FCVOpt implements the algorithm described in:

"Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms"
*Suraj Yerramilli and Daniel W. Apley*
Published in *Technometrics* (2025)
DOI: `10.1080/00401706.2025.2515926 <https://doi.org/10.1080/00401706.2025.2515926>`_

📚 Documentation Contents
------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   examples/01_Introduction_to_FCVOpt.ipynb
   

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
