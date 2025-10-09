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

    def _initialize(self, n_init: Optional[int] = None):
        """Initialize optimizer with random points or evaluate pending candidates.

        On first call, generates n_init random configurations and evaluates them on
        randomly selected folds. On subsequent calls, evaluates pending candidates
        from the last acquisition step.

        Args:
            n_init: Number of initial random points. Only used on first call.
        """
        if self.train_confs is None:
            # First call - initialize with random configurations and folds
            if n_init is None:
                n_init = len(self.config.quant_index) + 1

            self.config.seed(np.random.randint(2e+4))
            self.train_confs = list(self.config.latinhypercube_sample(n_init))

            # Initialize fold assignments
            if self.fold_initialization == 'random':
                self.train_folds = torch.randint(self.n_folds, (n_init, 1)).double()
            elif self.fold_initialization == 'stratified':
                self.train_folds = torch.from_numpy(stratified_sample(self.n_folds, n_init)).double().view(-1, 1)
            elif self.fold_initialization == 'two_folds':
                folds_choice = np.random.choice(self.n_folds, 2, replace=False)
                fold_1_samples = n_init // 2
                fold_0_samples = n_init - fold_1_samples
                self.train_folds = torch.tensor(
                    [folds_choice[0]] * fold_0_samples + [folds_choice[1]] * fold_1_samples
                ).double().view(-1, 1)

            # Evaluate initial design
            xs, ys, ts = [], [], []
            for conf, fold_idx in zip(self.train_confs, self.train_folds):
                fold_idx_int = fold_idx.int().item()
                x, y, t_eval = self._evaluate(conf, fold_idxs=[fold_idx_int])
                xs.append(x)
                ys.append(y)
                ts.append(t_eval)
                # Log each evaluation to MLflow with fold information
                self._log_eval(conf, x, y, t_eval, fold_idx=fold_idx_int)

            self.train_x = torch.tensor(xs).double()
            self.train_y = torch.tensor(ys).double()
            self.obj_eval_time = torch.tensor(ts).double()
        else:
            # Evaluate pending candidates from last acquisition
            if not hasattr(self, '_pending_candidates') or not self._pending_candidates:
                return

            # Use base class pending candidates if available, otherwise use FCVOpt-specific storage
            if hasattr(self, '_pending_candidates') and self._pending_candidates:
                # Use pending candidates from base class and corresponding folds
                next_confs_list = self._pending_candidates
                if hasattr(self, '_pending_folds'):
                    next_folds_list = self._pending_folds
                else:
                    # Fallback to stored folds_cand
                    next_folds_list = self.folds_cand[-1] if self.folds_cand else [0] * len(next_confs_list)
            else:
                # Fallback to FCVOpt-specific storage
                next_confs_list = self.confs_cand[-1] if self.confs_cand else []
                next_folds_list = self.folds_cand[-1] if self.folds_cand else []

            if not next_confs_list:
                return

            xs, ys, ts = [], [], []
            for i, (conf, fold_idx) in enumerate(zip(next_confs_list, next_folds_list)):
                x, y, t_eval = self._evaluate(conf, fold_idxs=[fold_idx])
                xs.append(x)
                ys.append(y)
                ts.append(t_eval)
                self.train_confs.append(conf)
                # Log each evaluation to MLflow with fold information
                self._log_eval(conf, x, y, t_eval, fold_idx=fold_idx)

            if xs:  # Only update if we have new evaluations
                self.train_x = torch.cat([self.train_x, torch.tensor(xs).double().to(self.train_x)], dim=0)
                self.train_y = torch.cat([self.train_y, torch.tensor(ys).double().to(self.train_y)], dim=0)
                self.obj_eval_time = torch.cat([self.obj_eval_time, torch.tensor(ts).double().to(self.obj_eval_time)], dim=0)

                # Update fold tracking
                new_folds = torch.tensor(next_folds_list).double().view(-1, 1).to(self.train_folds)
                self.train_folds = torch.cat([self.train_folds, new_folds], dim=0)

            # Clear pending candidates
            self._pending_candidates = None
            if hasattr(self, '_pending_folds'):
                self._pending_folds = None
    
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
    
    def _select_fold_indices(self, cand_confs):
        """Select fold indices for given candidate configurations.

        Args:
            cand_confs: List of candidate configurations.

        Returns:
            List[int]: Selected fold indices for each candidate.
        """
        num_candidates = len(cand_confs)
        total_num_folds = self.n_folds
        if self.n_repeats > 1 and self.train_folds.flatten().unique().shape[0] < self.n_folds:
            # consider sampling from other replicates only if all folds of the
            # first replicate have been evaluated at least once
            total_num_folds = self.n_folds * self.n_repeats

        fold_idxs = np.arange(total_num_folds)

        # Select folds for the candidates
        if self.fold_selection_criterion == 'random':
            selected_folds = np.random.default_rng(0).choice(
                fold_idxs, size=num_candidates, replace=True
            ).tolist()

        elif self.fold_selection_criterion == 'variance_reduction':
            selected_folds = []
            next_xs = np.row_stack([conf.get_array() for conf in cand_confs])

            for j, next_x in enumerate(next_xs):
                # shuffling to prevent ties among folds
                np.random.default_rng(j).shuffle(fold_idxs)
                fold_metrics = self.model._fold_selection_metric(
                    torch.from_numpy(next_x).view(1, -1), fold_idxs
                )
                selected_folds.append(fold_idxs[np.argmin(fold_metrics)])

        return selected_folds

    def _acquisition(self, i: int) -> None:
        """Propose next hyperparameter-fold pairs using acquisition and fold criterion.

        Extends the base class acquisition to add fold selection after candidate selection.

        Args:
            i: Current iteration number for logging.
        """
        # Use base class to select candidate configurations
        cand_confs = self._select_next_candidates(i)
        self.curr_conf_cand = cand_confs
        self._pending_candidates = cand_confs

        # Select folds for the candidates
        selected_folds = self._select_fold_indices(cand_confs)

        # Store folds in both formats for compatibility
        self.folds_cand.append(selected_folds)
        self._pending_folds = selected_folds

        # Log the iteration snapshot with complete information (including folds)
        self._log_iteration_snapshot(i)

    def _format_candidate_configs(self):
        """Format candidate configurations with fold information for logging.

        Returns:
            List: Formatted candidate configurations with fold information.
        """
        conf_cand_with_folds = []
        if self.curr_conf_cand and hasattr(self, '_pending_folds') and self._pending_folds:
            for conf, fold_idx in zip(self.curr_conf_cand, self._pending_folds):
                conf_cand_with_folds.append({
                    "config": dict(conf),
                    "fold_idx": int(fold_idx)
                })
        else:
            # Fallback to just configs without fold info
            conf_cand_with_folds = [dict(c) for c in (self.curr_conf_cand or [])]

        return conf_cand_with_folds

    def _log_eval(self, conf, x, y, eval_time, fold_idx=None):
        """Log a single evaluation with fold information as a JSON artifact.

        Extends the base class method to include fold/environment information
        for fractional cross-validation tracking.

        Args:
            conf: Configuration that was evaluated.
            x: Numeric array representation of the configuration.
            y: Objective function value.
            eval_time: Time taken to evaluate the configuration.
            fold_idx: The fold/environment index on which this configuration was evaluated.
        """
        kwargs = {}
        if fold_idx is not None:
            kwargs['fold_idx'] = int(fold_idx)
        super()._log_eval(conf, x, y, eval_time, **kwargs)

    @classmethod
    def restore_from_mlflow(
        cls,
        obj: Callable,
        run_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tracking_dir: Optional[str] = None,
        model_checkpoint: str = "latest",
        n_folds: int = 5,
        n_repeats: int = 1,
        fold_selection_criterion: str = 'variance_reduction',
        fold_initialization: str = 'random',
        **kwargs
    ) -> "FCVOpt":
        """Restore an FCVOpt instance from MLflow with fold information.

        This extends the base class restoration to also rebuild the training
        fold information from logged evaluations.

        Args:
            obj: Objective function to use for the restored optimizer.
            run_id: MLflow run ID to restore from.
            experiment_name: MLflow experiment name (used with run_name).
            run_name: MLflow run name (used with experiment_name).
            tracking_uri: MLflow tracking URI.
            tracking_dir: MLflow tracking directory (alternative to tracking_uri).
            model_checkpoint: Model checkpoint to load.
            n_folds: Number of folds in cross-validation.
            n_repeats: Number of independent repeats of K-fold CV.
            fold_selection_criterion: Strategy for selecting the next fold to evaluate.
            fold_initialization: Strategy for assigning folds in the initial random sample.
            **kwargs: Additional keyword arguments for BayesOpt.

        Returns:
            FCVOpt: Restored optimizer instance with fold information.
        """
        import tempfile
        import json
        import torch
        from mlflow.tracking import MlflowClient

        # First, restore using the base class method - this gives us a BayesOpt instance
        base_instance = super().restore_from_mlflow(
            obj=obj,
            run_id=run_id,
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tracking_dir=tracking_dir,
            model_checkpoint=model_checkpoint
        )

        # Create a new FCVOpt instance with the restored configuration
        fcv_instance = cls(
            obj=obj,
            config=base_instance.config,
            n_folds=n_folds,
            n_repeats=n_repeats,
            fold_selection_criterion=fold_selection_criterion,
            fold_initialization=fold_initialization,
            minimize=base_instance.minimize,
            acq_function=base_instance.acq_function,
            **kwargs
        )

        # Copy all the restored state from the base instance
        for attr in ['train_confs', 'train_x', 'train_y', 'obj_eval_time', '_n_evals',
                     'curr_conf_inc', 'curr_f_inc_obs', 'curr_f_inc_est',
                     'curr_fit_time', 'curr_acq_opt_time', 'curr_acq_val',
                     '_total_iterations', '_run_id', '_experiment_id', '_tracking_uri',
                     'model', 'initial_params', '_mlflow_initialized', '_client']:
            if hasattr(base_instance, attr):
                setattr(fcv_instance, attr, getattr(base_instance, attr))

        # Now reconstruct fold information from the evaluation files
        try:
            client = fcv_instance._client or MlflowClient()

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download evaluation files to extract fold information
                eval_artifacts = client.list_artifacts(fcv_instance._run_id, path="evals")
                eval_files = [item.path for item in eval_artifacts if item.path.endswith('.json')]

                if eval_files:
                    # Load evaluations to get fold information
                    evaluations = []
                    for eval_file in sorted(eval_files):
                        local_path = client.download_artifacts(fcv_instance._run_id, eval_file, dst_path=tmp_dir)
                        with open(local_path, 'r') as f:
                            evaluations.append(json.load(f))

                    # Extract fold information
                    folds = [eval_data.get("fold_idx", 0) for eval_data in evaluations]
                    fcv_instance.train_folds = torch.tensor(folds).double().view(-1, 1)
                else:
                    # Fallback: create dummy fold information
                    n_evals = len(fcv_instance.train_confs) if fcv_instance.train_confs else 0
                    fcv_instance.train_folds = torch.zeros(n_evals, 1).double()

        except Exception as e:
            print(f"Warning: Could not restore fold information: {e}")
            # Fallback: create dummy fold information
            n_evals = len(fcv_instance.train_confs) if fcv_instance.train_confs else 0
            fcv_instance.train_folds = torch.zeros(n_evals, 1).double()

        # Initialize other FCVOpt-specific attributes
        fcv_instance.folds_cand = []

        return fcv_instance