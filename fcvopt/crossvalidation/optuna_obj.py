import numpy as np
from .cvobjective import CVObjective
from ..configspace import ConfigurationSpace,CSH

from typing import Callable

def get_optuna_objective(cvobj:CVObjective,config:ConfigurationSpace) -> Callable:
    def optuna_obj(trial) -> float:
        optuna_config = {} 
        for hyp in config.get_hyperparameters():
            if isinstance(hyp,CSH.UniformFloatHyperparameter):
                optuna_config[hyp.name] = trial.suggest_float(hyp.name,hyp.lower,hyp.upper,log=hyp.log)
            elif isinstance(hyp,CSH.UniformIntegerHyperparameter):
                optuna_config[hyp.name] = trial.suggest_int(hyp.name,hyp.lower,hyp.upper,log=hyp.log)
            elif isinstance(hyp,CSH.CategoricalHyperparameter):
                optuna_config[hyp.name] = trial.suggest_categorical(hyp.name,hyp.choices)


        fold_idxs = np.random.choice(len(cvobj.train_test_splits))
        return cvobj.cvloss(params=optuna_config,fold_idxs=[fold_idxs])

    return optuna_obj