Configuration Spaces
=====================

The configspace module provides utilities for defining hyperparameter configuration spaces and working with the ConfigSpace library.

Overview
------------

Configuration spaces define the hyperparameter search space for optimization. FCVOpt uses the `ConfigSpace <https://automl.github.io/ConfigSpace/>`_ library to define:

* Hyperparameter types (integer, float, categorical, ordinal)
* Value ranges and constraints
* Conditional hyperparameters
* Forbidden configurations

Basic Usage
---------------

.. code-block:: python

   from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
   from fcvopt.configspace import get_sklearn_configspace

   # Manual configuration space definition
   config_space = ConfigurationSpace()
   config_space.add_hyperparameter(Integer("n_estimators", 10, 1000, default=100))
   config_space.add_hyperparameter(Float("max_features", 0.1, 1.0, default=1.0))
   config_space.add_hyperparameter(Categorical("criterion", ["gini", "entropy"]))

   # Or use predefined spaces for common models
   rf_space = get_sklearn_configspace("RandomForestClassifier")

Predefined Configuration Spaces
------------------------------------

FCVOpt provides predefined configuration spaces for common scikit-learn models:

.. code-block:: python

   from fcvopt.configspace import get_sklearn_configspace

   # Get configuration space for Random Forest
   rf_space = get_sklearn_configspace("RandomForestClassifier")

   # Get configuration space for SVM
   svm_space = get_sklearn_configspace("SVC")

   # Get configuration space for Gradient Boosting
   gb_space = get_sklearn_configspace("GradientBoostingClassifier")

Advanced Features
---------------------

Conditional Hyperparameters:

.. code-block:: python

   from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
   from ConfigSpace.conditions import EqualsCondition

   config_space = ConfigurationSpace()

   # Base hyperparameter
   kernel = Categorical("kernel", ["linear", "rbf", "poly"])
   config_space.add_hyperparameter(kernel)

   # Conditional hyperparameter (only active when kernel="rbf")
   gamma = Float("gamma", 1e-5, 1e-1, log=True)
   config_space.add_hyperparameter(gamma)
   config_space.add_condition(EqualsCondition(gamma, kernel, "rbf"))

Working with XGBoost:

.. code-block:: python

   from fcvopt.configspace import get_xgboost_configspace

   # XGBoost classification
   xgb_space = get_xgboost_configspace("XGBClassifier")

   # XGBoost regression
   xgb_reg_space = get_xgboost_configspace("XGBRegressor")

API Reference
-----------------

.. currentmodule:: fcvopt.configspace
.. automodule:: fcvopt.configspace
   :members: