Cross-Validation Objectives
============================

The cross-validation module provides convenient wrapper classes to perform K-fold cross-validation for various types of machine learning models. These objectives define the function that FCVOpt optimizes during hyperparameter tuning.

Overview
------------

Cross-validation objectives wrap machine learning models and define how to:

* Set up K-fold cross-validation splits
* Train models with given hyperparameters
* Evaluate performance on validation folds
* Handle model-specific requirements (early stopping, etc.)

Models Supported
----------------

* Scikit-learn models: Any sklearn estimator (RandomForest, SVM, etc.)
* XGBoost: With early stopping support
* Neural Networks: Multi-layer perceptrons and Tabular ResNet architectures
* Custom models: Extend the base CVObjective class

CVObjective Base Class
----------------------
.. autoclass:: CVObjective
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __call__


Scikit-learn Wrappers
---------------------

.. autoclass:: SklearnCVObj
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __call__ 

.. autoclass:: XGBoostCVObjEarlyStopping
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: MLPCVObj
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: ResNetCVObj
    :members:
    :undoc-members:
    :show-inheritance:


Optuna Wrapper
-----------------------------
.. autofunction:: fcvopt.crossvalidation.optuna_obj.get_optuna_objective
    :noindex:

Utility classes
-----------------------------

.. autoclass:: fcvopt.crossvalidation.mlp_cvobj.MLP
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fcvopt.crossvalidation.resnet_cvobj.TabularResNet
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fcvopt.crossvalidation.resnet_cvobj.ResNetBlock
    :members:
    :undoc-members:
    :show-inheritance:
