fcvopt.crossvalidation
======================

.. automodule:: fcvopt.crossvalidation
.. currentmodule:: fcvopt.crossvalidation

This module provides convience classes to perform K-fold cross-validation for various classes
of estimators.

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
