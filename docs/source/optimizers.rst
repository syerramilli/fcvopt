Optimizers
===========

The optimizers module contains the core optimization algorithms for hyperparameter tuning.

Overview
------------

FCVOpt provides two main optimizer classes:

* FCVOpt: The main fractional cross-validation optimizer using hierarchical Gaussian processes
* BayesOpt: Base Bayesian optimization framework for standard (full) cross-validation

API Reference
-----------------

.. automodule:: fcvopt.optimizers
.. currentmodule:: fcvopt.optimizers

BayesOpt Base Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: BayesOpt
   :members:
   :undoc-members:
   :show-inheritance:


FCVOpt Class
~~~~~~~~~~~~

.. autoclass:: FCVOpt
   :members:
   :undoc-members:
   :show-inheritance:

