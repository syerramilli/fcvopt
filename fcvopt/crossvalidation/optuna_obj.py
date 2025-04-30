import numpy as np
from .cvobjective import CVObjective
from ..configspace import ConfigurationSpace,CSH

from typing import Callable, List, Optional

def get_optuna_objective(cvobj:CVObjective, config:ConfigurationSpace, 
                         start_fold_idxs:Optional[List] = None,
                         rng_seed:Optional[int]=None) -> Callable:
    '''Utility function that returns the cross-validation objective
        for use with optuna.

    Args:
        cvobj: a :class:`CVObjective` object that defines the cross-validation objective
        config: a ConfigurationSpace object that defines the hyperparameter search space
        start_fold_idxs: a list of integers that define the starting fold indices for each
            trial. If None, a random fold is chosen for each trial at start. After the first
            len(start_fold_idxs) trials, the remaining trials will choose a random fold.
            If None, a random fold is chosen for the initial trials as well.

    Returns:
        A function that takes in a trial object from optuna and returns the validation
        loss at a randomly chosen fold for the given hyperparameter configuration.
    '''
    rng = np.random.RandomState(rng_seed)
    def optuna_obj(trial) -> float:
        optuna_config = {} 
        for hyp in list(config.values()):
            if isinstance(hyp,CSH.UniformFloatHyperparameter):
                optuna_config[hyp.name] = trial.suggest_float(hyp.name,hyp.lower,hyp.upper,log=hyp.log)
            elif isinstance(hyp,CSH.UniformIntegerHyperparameter):
                optuna_config[hyp.name] = trial.suggest_int(hyp.name,hyp.lower,hyp.upper,log=hyp.log)
            elif isinstance(hyp,CSH.CategoricalHyperparameter):
                optuna_config[hyp.name] = trial.suggest_categorical(hyp.name,hyp.choices)

        if start_fold_idxs is not None and trial.number < len(start_fold_idxs):
            fold_idxs = start_fold_idxs[trial.number]
        else:
            fold_idxs = rng.choice(len(cvobj.train_test_splits))
        return cvobj.cvloss(params=optuna_config,fold_idxs=[fold_idxs])

    return optuna_obj