import numpy as np
import torch
from .bayes_opt import BayesOpt
from ..models import HGP

from ..configspace import ConfigurationSpace
from ConfigSpace import Configuration
from ..util.samplers import stratified_sample
from typing import Callable, List ,Optional 
import joblib

class FCVOpt(BayesOpt):
    """Fractional cross-validation for hyperparameter optimization
    
    Implements the fractional CV approach from "Fractional cross-validation for optimizing
    hyperparameters of supervised learning algorithms." This method uses a hierarchical
    Gaussian process (HGP) model to exploit the correlation of single-fold out-of-sample
    errors across hyperparameter configurations, enabling efficient Bayesian
    optimization with only a fraction of the K folds evaluated per configuration.

    Rather than performing full K-fold CV at every candidate point, FCVOpt:
      - Employs a hierarchical GP that models both fold-wise and hyperparameter-wise
        covariance structures
      - Evaluates only one fold (or a small subset of folds) for most configurations,
        drastically reducing computation
      - Selects folds adaptively based on variance reduction or random sampling

    Args:
        obj: Objective function that takes a hyperparameter configuration dict and returns
            a scalar cross-validation error for a given fold index list.
        config: Hyperparameter search space.
        n_folds: Number of folds in standard K-fold cross-validation.
        n_repeats: Number of independent repeats of K-fold CV; used to expand the fold index 
            set. Defaults to 1.
        fold_selection_criterion: Strategy for selecting the next fold to evaluate:
            - 'variance_reduction': choose the fold that minimizes predictive variance via
              HGP._fold_selection_metric
            - 'random': choose folds uniformly at random
            Defaults to 'variance_reduction'.
        fold_initialization: Strategy for assigning folds in the initial random sample of configurations:
            - 'random': sample folds uniformly at random
            - 'stratified': use stratified sampling across folds via
              :func:`fcvopt.util.samplers.stratified_sample`
            - 'two_folds': randomly pick two distinct folds and split samples between them
            Defaults to 'random'.
        minimize: If True, minimizes the cross-validation error; otherwise maximizes. Defaults to True.
        acq_function: Acquisition function to use. One of {'LCB', 'KG'}. Note that 'EI' is not supported and will 
            raise a RuntimeError. Defaults to 'LCB'.
        **kwargs: Additional keyword arguments passed to :class:`.bayes_opt.BayesOpt`:
            `acq_function_options`, `batch_acquisition`, `acquisition_q`, `verbose`,
            `save_iter`, `save_dir`, and `n_jobs`.

    References:
        - :class:`fcvopt.models.HGP`
    """
    def __init__(
        self,
        obj:Callable,
        config:ConfigurationSpace,
        n_folds:int,
        n_repeats:int=1,
        fold_selection_criterion:str='variance_reduction',
        fold_initialization:str='random',
        minimize:bool=True,
        acq_function:str='LCB',
        **kwargs
    ):
        if acq_function == 'EI':
            raise RuntimeError('Expected improvment not implemented for FCVOPT')

        super().__init__(
            obj=obj,config=config,minimize=minimize,acq_function=acq_function,**kwargs
        )
        
        # fold indices and candidates not present in BayesOpt
        # TODO: add checks for the validity of fold_selection criterion
        self.fold_selection_criterion = fold_selection_criterion
        self.fold_initialization = fold_initialization
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.train_folds = None
        self.folds_cand = []

    @property
    def stats_keys(self) -> List:
        return [
            'f_inc_obs','f_inc_est','acq_vec',
            'confs_inc','confs_cand','folds_cand',
            'fit_time','acqopt_time','obj_eval_time',
        ]

    @property
    def data_keys(self)->List:
        return ['train_x','train_y','train_folds']

    def _initialize(self,n_init:Optional[int]=None):
        if self.train_confs is None:
            if n_init is None:
                n_init = len(self.config.quant_index) + 1
            
            self.config.seed(np.random.randint(2e+4))
            self.train_confs = self.config.latinhypercube_sample(n_init)
            # initialize fold to be from among the first replicate    
            #self.train_folds = torch.randint(self.n_folds,(1,1)).repeat(n_init,1).double()
            if self.fold_initialization == 'random':
                self.train_folds = torch.randint(self.n_folds,(n_init,1)).double()
            elif self.fold_initialization == 'stratified':
                self.train_folds = torch.from_numpy(stratified_sample(self.n_folds,n_init)).double().view(-1,1)
            elif self.fold_initialization == 'two_folds':
                folds_choice = np.random.choice(self.n_folds, 2, replace=False)
                fold_1_samples = n_init // 2
                fold_0_samples = n_init - fold_1_samples

                self.train_folds = torch.tensor(
                    [folds_choice[0]] * fold_0_samples + [folds_choice[1]] * fold_1_samples
                ).double().view(-1, 1)

            for conf,fold_idx in zip(self.train_confs,self.train_folds):
                x,y,eval_time = self._evaluate(conf,fold_idxs=fold_idx.int().tolist())
                self.train_x.append(x)
                self.train_y.append(y)
                self.obj_eval_time.append(eval_time)
            
            self.train_x = torch.from_numpy(np.row_stack(self.train_x)).double()
            self.train_y = torch.tensor(self.train_y).double()
        else:
            # algorithm has been run previously
            # evaluate the next candidate 
            next_confs_list = self.confs_cand[-1]
            next_folds_list = self.folds_cand[-1]
            
            evaluations = self._evaluate_confs(next_confs_list, next_folds_list)
            for i,(next_x,next_y,eval_time) in enumerate(evaluations):
                self.train_confs.append(next_confs_list[i])
                self.train_x = torch.cat([
                    self.train_x,torch.tensor(next_x).to(self.train_x).reshape(1,-1)]
                )
                self.train_y = torch.cat([self.train_y,torch.tensor([next_y]).to(self.train_y)])
            
                self.train_folds = torch.cat([
                    self.train_folds,torch.tensor(next_folds_list[i]).to(self.train_folds).reshape(1,-1)]
                )

                self.obj_eval_time.append(eval_time)
    
    def _evaluate_confs(
            self,
            confs_list:List[Configuration],
            folds_list:List[int],
            **kwargs
        ):
        if self.n_jobs > 1 and len(confs_list) > 1:
            # enable parallel evaulations
            evaluations = joblib.Parallel(n_jobs=self.n_jobs,verbose=0)(
                joblib.delayed(self._evaluate)(conf,fold_idxs=[fold_idx],**kwargs) \
                    for conf,fold_idx in zip(confs_list,folds_list)
            )
        else:
            # can add logging here
            evaluations = [None]*len(confs_list)
            for i in range(len(confs_list)):
                evaluations[i] = self._evaluate(confs_list[i],fold_idxs=[folds_list[i]],**kwargs)
        
        return evaluations

    def _construct_model(self) -> HGP:
        return HGP(
            train_x = (self.train_x,self.train_folds),
            train_y = self.sign_mul*self.train_y
        ).double()
    
    def _acquisition(self) -> None:
        """Propose next hyperparameter-fold pairs using acquisition and fold criterion.

        1. Call `BayesOpt._acquisition` to get next hyperparameter candidates.
        2. Select fold(s) per candidate based on `fold_selection_criterion`, either
           minimizing predictive variance (via HGP) or random sampling.
        """
        # acquisition for x is the same as BayesOpt
        super()._acquisition()

        num_candidates = 1 if not self.batch_acquisition else self.acquisition_q
        
        total_num_folds = self.n_folds
        if self.n_repeats > 1 and self.train_folds.flatten().unique().shape[0] < self.n_folds:
            # consider sampling from other replicates only if all folds of the 
            # first replicate have been evaluated at least once
            total_num_folds = self.n_folds*self.n_repeats

        fold_idxs = np.arange(total_num_folds)

        if self.fold_selection_criterion == 'random':
            self.folds_cand.append(
                np.random.default_rng(0).choice(
                    fold_idxs,size=num_candidates,replace=True
                ).tolist()
            )

        elif self.fold_selection_criterion == 'variance_reduction':
            folds = []
            next_xs = np.row_stack([
                conf.get_array() for conf in self.confs_cand[-1]
            ])
            
            for i, next_x in enumerate(next_xs):
                # shuffling to prevent ties among folds
                np.random.default_rng(i).shuffle(fold_idxs)
                fold_metrics = self.model._fold_selection_metric(
                    torch.from_numpy(next_x).view(1,-1),fold_idxs
                )
                folds.append(fold_idxs[np.argmin(fold_metrics)])
            
            self.folds_cand.append(folds)