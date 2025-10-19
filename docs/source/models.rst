Gaussian Process Models
========================

The models module contains Gaussian Process implementations used for surrogate modeling in Bayesian optimization.

Overview
------------

FCVOpt uses Gaussian Processes to model the relationship between hyperparameters and cross-validation performance. Two main model types are available:

* GPR: Standard single-task Gaussian Process regression
* HGP: Hierarchical Gaussian Process for modeling fold-wise correlations

Key Concepts
----------------

Hierarchical Gaussian Processes (HGP):
The core innovation in FCVOpt. Instead of treating each CV fold independently, HGP models the correlation structure between folds, enabling accurate prediction of performance on unevaluated folds.

Multi-task Learning:
HGP treats each CV fold as a separate "task" and learns correlations between tasks, dramatically reducing the number of fold evaluations needed.

Kernel Functions:
Both models support various kernel functions for modeling different types of relationships:

* Matern kernels for smooth functions
* Constant kernels for bias terms
* Hamming kernels for categorical variables
* Multi-task kernels for fold correlations

Usage in FCVOpt
-------------------

These models are used internally by the optimizers and typically don't need to be instantiated directly:

.. code-block:: python

   # HGP is used automatically when using FCVOpt
   from fcvopt.optimizers import FCVOpt

   optimizer = FCVOpt(cv_obj, config_space, acq='kg')
   # Internally uses HGP to model fold correlations

   # Standard GP is used with BayesOpt for full CV
   from fcvopt.optimizers import BayesOpt

   optimizer = BayesOpt(cv_obj, config_space, acq='lcb')
   # Internally uses GPR for standard modeling

Advanced Usage
------------------

For custom implementations or research purposes, the models can be used directly:

.. code-block:: python

   from fcvopt.models import HGP
   import torch

   # Create hierarchical GP model
   model = HGP(
       train_X=train_configs,     # Hyperparameter configurations
       train_Y=train_scores,      # CV scores for each fold
       num_folds=5,               # Number of CV folds
       fold_indices=fold_ids      # Which fold each observation belongs to
   )

   # Train the model
   model.train()

   # Make predictions on new configurations
   posterior = model.posterior(test_configs)
   mean = posterior.mean
   variance = posterior.variance

API Reference
-----------------

.. automodule:: fcvopt.models
.. currentmodule:: fcvopt.models

Hierarchical Gaussian Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HGP
   :members:
   :undoc-members:
   :show-inheritance:

Standard Gaussian Process Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GPR
   :members:
   :undoc-members:
   :show-inheritance: