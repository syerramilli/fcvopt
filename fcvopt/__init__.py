"""
FCVOpt: Fractional Cross-Validation for Hyperparameter Optimization

A Python package for efficient hyperparameter tuning using fractional
cross-validation with hierarchical Gaussian processes.
"""

from fcvopt.optimizers import BayesOpt, FCVOpt
from fcvopt.crossvalidation import CVObjective, SklearnCVObj
from fcvopt.configspace import ConfigurationSpace

__all__ = [
    "__version__",
    "BayesOpt",
    "FCVOpt",
    "CVObjective",
    "SklearnCVObj",
    "ConfigurationSpace",
]
