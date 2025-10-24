Configuration Spaces
=====================


Configuration spaces define the hyperparameter search space for optimization. FCVOpt uses a thin wrapper around
`ConfigSpace <https://automl.github.io/ConfigSpace/>`_ library. 

Basic Usage
---------------

.. code-block:: python

   from fcvopt.configspace import ConfigurationSpace
   from ConfigSpace import Float, Integer

   # Manual configuration space definition
   config_space = ConfigurationSpace()
   config.add([
      Integer('n_estimators', bounds=(50, 1000), log=True),
      Integer('max_depth', bounds=(1, 15), log=True),
      Float('max_features', bounds=(0.01, 1.0), log=True),
      Integer('min_samples_split', bounds=(2, 200), log=True)
   ])

API Reference
-----------------

.. currentmodule:: fcvopt.configspace
.. automodule:: fcvopt.configspace
   :members: