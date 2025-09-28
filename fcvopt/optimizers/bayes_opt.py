import os, time, json, warnings, joblib, random
from collections import OrderedDict
from typing import Optional, Dict, List, Tuple, Callable, Any

import numpy as np
import torch
import gpytorch
import mlflow
from mlflow.tracking import MlflowClient
import tempfile

from ConfigSpace import Configuration
import ConfigSpace as CS
from ..configspace import ConfigurationSpace
from ..fit.mll_scipy import fit_model_scipy
from ..models import GPR

from botorch.acquisition import (
    ExpectedImprovement, qExpectedImprovement,
    UpperConfidenceBound, qUpperConfidenceBound,
    qKnowledgeGradient
)
from botorch.sampling import SobolQMCNormalSampler

from .optimize_acq import _optimize_botorch_acqf

def _set_seed(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class BayesOpt:
    """Bayesian Optimization for optimizing a given objective function.

    This class implements a Bayesian optimization loop that iteratively fits a Gaussian
    Process model to the objective function and proposes new configurations to evaluate
    via an acquisition function. The optimizer supports MLflow tracking for experiment
    management and can be used with various acquisition functions and batch optimization.

    Args:
        obj: The objective function mapping a configuration dict to a scalar value.
        config: The search space configuration.
        minimize: If True, minimizes the objective; otherwise maximizes it. Defaults to True.
        acq_function: Acquisition function to use. One of {'EI', 'LCB', 'KG'}. Defaults to 'EI'.
        acq_function_options: Additional keyword arguments passed to the acquisition function constructor.
            Defaults to None.
        batch_acquisition: If True, a batch of configurations (the number specified by `acquisition_q`)
            is selected for each iteration. Defaults to False.
        acquisition_q: Number of points in each proposed batch when `batch_acquisition` is True.
            Defaults to 1.
        verbose: Verbosity level to print to console; 0=no output, 1=summary at end, 2=detailed
            per-iteration log. Defaults to 1.
        n_jobs: Number of parallel jobs for objective evaluation and model fitting. Use -1 to utilize
            all available CPU cores. Defaults to 1.
        seed: Random seed for reproducibility. Defaults to None.
        tracking_uri: MLflow tracking URI (e.g., "file:/abs/path" or "http://host:5000").
            Defaults to None.
        tracking_dir: Directory for MLflow tracking (e.g., "./results"). Gets converted to absolute
            file URI. Defaults to None.
        experiment: MLflow experiment name. Defaults to None.
        run_name: MLflow run name. Defaults to None.
        model_checkpoint_freq: Model checkpointing frequency. Save checkpoint every N iterations.
            1 = every iteration, 5 = every 5 iterations. Always saves final iteration.
            Defaults to 1.

    Examples:
        Basic usage:

        >>> from fcvopt.optimizers import BayesOpt
        >>> from fcvopt.configspace import ConfigurationSpace
        >>>
        >>> # Define objective function
        >>> def objective(config):
        ...     return config['x']**2 + config['y']**2
        >>>
        >>> # Define search space
        >>> import ConfigSpace as CS
        >>> cs = ConfigurationSpace()
        >>> cs.add(CS.Float('x', bounds=(-5.0, 5.0)))
        >>> cs.add(CS.Float('y', bounds=(-5.0, 5.0)))
        >>>
        >>> # Run optimization
        >>> bo = BayesOpt(obj=objective, config=cs)
        >>> results = bo.run(n_iter=10)
        >>> print(f"Best config: {results['conf_inc']}")
        >>> print(f"Best value: {results['f_inc_obs']}")

        Continuing optimization:

        >>> # Continue with more iterations
        >>> results = bo.run(n_iter=5)  # MLflow run will be reactivated
    """
    def __init__(
        self,
        obj: Callable,
        config: ConfigurationSpace,
        minimize: bool = True,
        acq_function: str = 'EI',
        acq_function_options: Optional[Dict] = None,
        batch_acquisition: bool = False,
        acquisition_q: int = 1,
        verbose: int = 1,
        n_jobs: int = 1,
        seed: Optional[int] = None,
        tracking_uri: Optional[str] = None,      # e.g., "file:/abs/path"or "http://host:5000"
        tracking_dir: Optional[str] = None,      # e.g., "./results" (turns into file:/ABS/PATH)
        experiment: Optional[str] = None,    # MLflow experiment name 
        run_name: Optional[str] = None,
        model_checkpoint_freq: int = 1, # Save checkpoint every N iterations
    ):
        # ---- core config ----
        self.obj = obj
        self.config = config
        self.minimize = minimize
        self.sign_mul = -1 if minimize else 1
        self.acq_function = acq_function
        self.acquisition_function_options = acq_function_options or {}
        self.batch_acquisition = batch_acquisition
        self.acquisition_q = int(acquisition_q)
        self.n_jobs = n_jobs
        self.verbose = verbose

        # ---- BO state (training data kept in memory) ----
        self.model: Optional[GPR] = None
        self.train_confs: Optional[List[Configuration]] = None
        self.train_x: Optional[torch.Tensor] = None
        self.train_y: Optional[torch.Tensor] = None
        self.obj_eval_time: Optional[torch.Tensor] = None
        self.initial_params: Optional[OrderedDict] = None

        # current-iteration snapshot
        self.curr_conf_inc: Optional[Configuration] = None
        self.curr_conf_cand: Optional[List[Configuration]] = None
        self.curr_f_inc_obs: Optional[float] = None
        self.curr_f_inc_est: Optional[float] = None
        self.curr_fit_time: Optional[float] = None
        self.curr_acq_opt_time: Optional[float] = None
        self.curr_acq_val: Optional[float] = None

        self._pending_candidates: Optional[List[Configuration]] = None
        self._n_evals: int = 0

        self._model_checkpoint_freq = model_checkpoint_freq
        self._mlflow_initialized = False
        self._total_iterations = 0  # Track total iterations across multiple runs
        self._run = None
        self._client = None
        self._run_id = None
        self._experiment_id = None
        self._tracking_uri = None

        # Store MLflow config for lazy initialization
        self._mlflow_config = {
            'tracking_uri': tracking_uri,
            'tracking_dir': tracking_dir,
            'experiment': experiment,
            'run_name': run_name,
            'seed': seed
        }

        _set_seed(seed)

    # ========================== PUBLIC API ========================== #
    def run(self,n_iter:int,n_init:Optional[int]=None) -> Dict:
        """Run BO for exactly `n_iter` acquisition steps.

        If this is the first call, initializes the optimizer with n_init random points.
        If called again on the same instance, continues optimization from where it left off.

        Args:
            n_iter (int): Number of acquisition iterations to run.
            n_init (int, optional): Number of initial random points. Only used on first call.
                If None on first call, defaults to len(config.quant_index) + 1.
                Ignored on subsequent calls.

        Returns:
            Dict: Results containing incumbent configuration and objective values.
        """
        # Check if this is a continuation
        is_continuation = self.train_confs is not None

        # Initialize MLflow on first call, or reactivate on continuation
        if not self._mlflow_initialized:
            self._initialize_mlflow()
        elif is_continuation:
            # Reactivate the existing run for continuation
            self._ensure_mlflow_active()
        if is_continuation and n_init is not None:
            if self.verbose >= 1:
                print(f"Warning: n_init={n_init} ignored for continuation run")
            n_init = None  # Ignore n_init for continuation

        output_header = '%6s %12s %12s %12s' % ('iter', 'f_inc_obs', 'f_inc_est', 'acq_val')

        for i in range(n_iter):
            # Use global iteration counter for logging
            global_iter = self._total_iterations + i

            # initialize (i==0) or evaluate previous candidates (i>0)
            self._initialize(n_init)

            # fit GP and update incumbent
            self._fit_model_and_find_inc(global_iter)

            # choose next candidates via acquisition
            self._acquisition(global_iter)

            # checkpoint model based on frequency or if final iteration
            is_final = (i == n_iter - 1)
            self._save_and_log_model_state_iter(global_iter, is_final=is_final)

            # console output
            if self.verbose >= 2:
                if global_iter % 10 == 0:
                    print(output_header)
                print('%6i %12.3e %12.3e %12.3e' %
                      (global_iter, self.curr_f_inc_obs, self.curr_f_inc_est, self.curr_acq_val))

        # Update total iterations counter
        self._total_iterations += n_iter

        # summary
        if self.verbose >= 1:
            status_msg = "after continuation" if is_continuation else "at termination"
            print(f'\nNumber of candidates evaluated.....: {len(self.train_confs)}')
            print(f'Observed obj at incumbent..........: {self.curr_f_inc_obs:.6g}')
            print(f'Estimated obj at incumbent.........: {self.curr_f_inc_est:.6g}')
            print(f'\nIncumbent {status_msg}:\n', self.curr_conf_inc)
            print(f'\nCandidate(s) {status_msg}:\n', self.curr_conf_cand)

        # Log current metrics
        self._ensure_mlflow_active()
        mlflow.log_metrics({
            "current_f_inc_obs": float(self.curr_f_inc_obs),
            "current_f_inc_est": float(self.curr_f_inc_est),
            "total_evals": int(len(self.train_confs)),
        })

        # For first run, end the MLflow run. For subsequent runs, continue the same run.
        if not is_continuation:
            mlflow.log_metrics({
                "final_f_inc_obs": float(self.curr_f_inc_obs),
                "final_f_inc_est": float(self.curr_f_inc_est),
            })
            mlflow.set_tag("status", "completed")
            mlflow.end_run()

        results = OrderedDict()
        results['conf_inc'] = self.curr_conf_inc
        results['f_inc_obs'] = self.curr_f_inc_obs
        results['f_inc_est'] = self.curr_f_inc_est
        return results
    
    def optimize(self, n_trials:int, n_init:Optional[int]=None) -> Dict:
        """Run Bayesian optimization for a specified number of trials in this call.

        This method treats ``n_trials`` as the number of evaluations to perform
        **in this specific call**, making it suitable for both initial runs and
        continuation runs from restored optimizers.

        For initial runs, it evaluates ``n_trials`` total configurations:
        ``n_init`` random initializations followed by ``n_trials - n_init``
        acquisition-guided evaluations.

        For continuation runs (when optimization has already been started), it
        performs exactly ``n_trials`` additional evaluations via acquisition,
        ignoring the ``n_init`` parameter.

        Args:
            n_trials (int):
                Number of objective evaluations to perform in this call.
                For initial runs, includes both random and acquisition-guided trials.
                For continuation runs, specifies additional acquisition-guided trials.
                Must be positive.
            n_init (int, optional):
                Number of initial random configurations for the first call only.
                If ``None`` on initial run, defaults to ``len(self.config.quant_index) + 1``.
                Ignored and warned about for continuation runs.

        Returns:
            Dict:
                Ordered results identical to :meth:`run`, with keys:
                - ``'conf_inc'``: incumbent configuration,
                - ``'f_inc_obs'``: observed objective at incumbent,
                - ``'f_inc_est'``: model-estimated objective at incumbent.

        Raises:
            ValueError: If ``n_trials`` is not positive or invalid ``n_init`` for initial runs.

        Examples:
            >>> # Initial run: 15 total evaluations (3 random + 12 acquisitions)
            >>> results = bo.optimize(n_trials=15, n_init=3)
            >>> len(bo.train_confs)  # Should be 15

            >>> # Continuation: 10 more evaluations (all acquisitions)
            >>> results = bo.optimize(n_trials=10)  # n_init ignored
            >>> len(bo.train_confs)  # Should be 25 (15 + 10)

            >>> # Another continuation: 5 more evaluations
            >>> results = bo.optimize(n_trials=5)
            >>> len(bo.train_confs)  # Should be 30 (25 + 5)
        """
        if n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got {n_trials}")

        # Check if this is a continuation run
        is_continuation = self.train_confs is not None

        if is_continuation:
            # For continuation runs, perform n_trials additional acquisitions
            if n_init is not None and self.verbose >= 1:
                print(f"Warning: n_init={n_init} ignored for continuation run")

            # Perform n_trials acquisition steps
            n_iter = n_trials
            # Call run and then evaluate any remaining pending candidates to ensure we get exactly n_trials evaluations
            initial_count = len(self.train_confs)
            results = self.run(n_iter=n_iter, n_init=None)

            # Check if we have pending candidates that need evaluation to reach exactly n_trials more
            if len(self.train_confs) - initial_count < n_trials and self._pending_candidates:
                # Evaluate the final pending candidates
                self._initialize(n_init=None)
                # Update incumbent after final evaluation
                self._fit_model_and_find_inc(self._total_iterations)
                # Update results
                results['conf_inc'] = self.curr_conf_inc
                results['f_inc_obs'] = self.curr_f_inc_obs
                results['f_inc_est'] = self.curr_f_inc_est

            return results
        else:
            # Initial run: use original logic
            if n_init is None:
                n_init = len(self.config.quant_index) + 1

            if not (1 <= n_init <= n_trials):
                raise ValueError(f"n_init must be in [1, {n_trials}], got {n_init!r}")

            # compute how many acquisitions to perform after initialization
            n_iter = n_trials - n_init + 1
            return self.run(n_iter=n_iter, n_init=n_init)

    def end_run(self):
        """Manually end the MLflow run.

        Logs final metrics and sets the run status to completed before ending the run.
        This method is useful when you want to explicitly finish tracking before the
        optimizer goes out of scope.

        Note:
            This method is automatically called when using the optimizer as a context manager
            or when the first call to run() completes. Subsequent calls to run() will
            reactivate the same MLflow run.
        """
        if self._mlflow_initialized and mlflow.active_run() is not None:
            # Log final metrics when manually ending
            if hasattr(self, 'curr_f_inc_obs') and self.curr_f_inc_obs is not None:
                mlflow.log_metrics({
                    "final_f_inc_obs": float(self.curr_f_inc_obs),
                    "final_f_inc_est": float(self.curr_f_inc_est),
                })
            mlflow.set_tag("status", "completed")
            mlflow.end_run()

    def get_optimization_results(self) -> List[Dict[str, Any]]:
        """Retrieve detailed optimization results for all iterations.

        Returns a comprehensive summary of the optimization process, including
        incumbent configurations, observed values, and model predictions with
        uncertainty estimates for each iteration.

        Returns:
            List[Dict[str, Any]]: List of dictionaries, one per iteration, with keys:
                - 'iteration' (int): Iteration number (0-based)
                - 'incumbent_config' (dict): Best configuration found so far
                - 'observed_value' (float): Actual observed objective value at incumbent
                - 'predicted_value' (float): Model's predicted value at incumbent
                - 'predicted_std' (float): Model's prediction uncertainty (standard deviation)

        Raises:
            RuntimeError: If no optimization has been performed yet or model is unavailable.

        Examples:
            >>> bo = BayesOpt(obj=objective, config=config_space)
            >>> bo.run(n_iter=5)
            >>> results = bo.get_optimization_results()
            >>>
            >>> # Access results for each iteration
            >>> for result in results:
            ...     print(f"Iteration {result['iteration']}:")
            ...     print(f"  Best config: {result['incumbent_config']}")
            ...     print(f"  Observed: {result['observed_value']:.4f}")
            ...     print(f"  Predicted: {result['predicted_value']:.4f} ± {result['predicted_std']:.4f}")
            ...
            >>> # Get final result
            >>> final_result = results[-1]
            >>> best_config = final_result['incumbent_config']
            >>> best_value = final_result['observed_value']
        """
        import warnings

        if not hasattr(self, 'train_confs') or not self.train_confs:
            raise RuntimeError("No optimization performed yet. Call run() first.")

        if self.model is None:
            raise RuntimeError("Model is not available. Ensure optimization has been performed.")

        # Prepare results list
        results = []

        # Get iteration data from MLflow artifacts if available
        iteration_data = self._get_iteration_data_from_artifacts()

        # If no iteration data from artifacts, construct from current state
        if not iteration_data:
            iteration_data = self._construct_iteration_data_from_state()

        # Generate predictions with uncertainty for each iteration's incumbent
        for i, iter_data in enumerate(iteration_data):
            incumbent_config = iter_data['incumbent_config']
            observed_value = iter_data['observed_value']

            # Convert config to model input format
            if isinstance(incumbent_config, dict):
                config_array = self._config_dict_to_array(incumbent_config)
            else:
                # Assume it's already a Configuration object
                config_array = incumbent_config.get_array()

            # Get model prediction with uncertainty
            try:
                with warnings.catch_warnings(), torch.no_grad():
                    warnings.simplefilter(action='ignore', category=gpytorch.utils.warnings.GPInputWarning)

                    # Prepare input tensor
                    x_tensor = torch.from_numpy(config_array).double().unsqueeze(0)

                    # Get prediction with standard deviation
                    pred_mean, pred_std = self.model.predict(x_tensor, return_std=True)
                    predicted_mean = float(self.sign_mul * pred_mean.item())
                    predicted_std = float(pred_std.item())

            except Exception as e:
                # Fallback: use observed value as prediction with zero uncertainty
                predicted_mean = observed_value
                predicted_std = 0.0
                if hasattr(self, 'verbose') and self.verbose >= 1:
                    print(f"Warning: Could not get model prediction for iteration {i}: {e}")

            # Create result dictionary with intuitive names
            result = {
                'iteration': i,
                'incumbent_config': incumbent_config if isinstance(incumbent_config, dict) else dict(incumbent_config),
                'observed_value': float(observed_value),
                'predicted_value': predicted_mean,
                'predicted_std': predicted_std
            }

            results.append(result)

        return results

    def __enter__(self):
        """Context manager entry.

        Returns:
            BayesOpt: The optimizer instance for use in the with statement.

        Examples:
            >>> with BayesOpt(obj=objective, config=cs) as bo:
            ...     results = bo.run(n_iter=10)
            ...     # MLflow run automatically ended when exiting context
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically end MLflow run.

        Args:
            exc_type: Exception type (if any).
            exc_val: Exception value (if any).
            exc_tb: Exception traceback (if any).

        Returns:
            bool: False to propagate any exceptions.
        """
        del exc_type, exc_val, exc_tb  # Silence unused parameter warnings
        self.end_run()
        return False

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
    ) -> "BayesOpt":
        """Restore a BayesOpt instance from MLflow run artifacts to continue optimization.

        This method reconstructs the complete state of a previous optimization run from
        MLflow logs, allowing you to seamlessly continue optimization where you left off.
        The restored instance will use the same experiment and run configuration to
        maintain continuity.

        Args:
            obj: Objective function for the restored optimizer.
            run_id: Specific MLflow run ID to restore from. If provided, takes precedence
                over experiment_name/run_name.
            experiment_name: Name of the MLflow experiment to search for the run.
                Used with run_name to identify the run to restore.
            run_name: Name of the specific run to restore within the experiment.
                Used with experiment_name to identify the run to restore.
            tracking_uri: MLflow tracking URI (e.g., "file:/path" or "http://host:5000").
                Cannot be used with tracking_dir.
            tracking_dir: Directory for MLflow tracking (e.g., "./results"). Gets converted
                to absolute file URI. Cannot be used with tracking_uri.
            model_checkpoint: Model checkpoint to load. Either "latest" for the most
                recent checkpoint, or a specific filename.

        Returns:
            BayesOpt: Restored optimizer instance ready to continue optimization.

        Raises:
            RuntimeError: If run cannot be found or required artifacts are missing.
            ValueError: If neither run_id nor (experiment_name, run_name) is provided.

        Examples:
            >>> # Restore by run ID with tracking_uri
            >>> bo = BayesOpt.restore_from_mlflow(
            ...     obj=objective_function,
            ...     run_id="abc123",
            ...     tracking_uri="file:/path/to/mlruns"
            ... )
            >>> results = bo.run(n_iter=5)  # Continue optimization

            >>> # Restore by experiment and run name with tracking_dir
            >>> bo = BayesOpt.restore_from_mlflow(
            ...     obj=objective_function,
            ...     experiment_name="BayesOpt",
            ...     run_name="optimization_run_1",
            ...     tracking_dir="./my_results"
            ... )
            >>> results = bo.run(n_iter=10)  # Continue optimization

        Note:
            The restored instance will:
            - Reconstruct all training data and state from MLflow artifacts
            - Reuse the same experiment and run configuration
            - Continue with the same MLflow run for seamless tracking
            - Load the model checkpoint for warm-starting
        """
        # Validate arguments
        if not run_id and not (experiment_name and run_name):
            raise ValueError("Either run_id or both experiment_name and run_name must be provided")

        if tracking_uri and tracking_dir:
            raise ValueError("Cannot specify both tracking_uri and tracking_dir. Use one or the other.")

        # Set tracking URI, converting tracking_dir if needed
        def _ensure_dir_exists(path: str):
            os.makedirs(path, exist_ok=True)
            return path

        if tracking_uri:
            if tracking_uri.startswith("file:"):
                local_path = tracking_uri[len("file:"):]
                _ensure_dir_exists(os.path.abspath(local_path))
            mlflow.set_tracking_uri(tracking_uri)
        elif tracking_dir:
            base_dir = os.path.abspath(tracking_dir)
            _ensure_dir_exists(base_dir)
            resolved_tracking_uri = f"file:{base_dir}"
            mlflow.set_tracking_uri(resolved_tracking_uri)
            tracking_uri = resolved_tracking_uri  # Store for later use

        client = MlflowClient()

        # Find the run
        if run_id:
            try:
                run = client.get_run(run_id)
            except Exception as e:
                raise RuntimeError(f"Run with ID '{run_id}' not found: {e}")
        else:
            # Search for run by experiment and run name
            try:
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    raise RuntimeError(f"Experiment '{experiment_name}' not found")

                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{run_name}'"
                )

                if not runs:
                    raise RuntimeError(f"Run '{run_name}' not found in experiment '{experiment_name}'")

                if len(runs) > 1:
                    raise RuntimeError(f"Multiple runs found with name '{run_name}' in experiment '{experiment_name}'")

                run = runs[0]
                run_id = run.info.run_id

            except Exception as e:
                raise RuntimeError(f"Error finding run: {e}")

        # Extract run info and parameters
        run_info = run.info
        run_params = run.data.params
        run_tags = run.data.tags

        # Create temporary directory for artifacts
        tmp_dir = os.path.join(tempfile.gettempdir(), f"mlflow_restore_{run_id}")
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            # Download and load configuration space
            config_path = os.path.join(tmp_dir, "config_space.json")
            client.download_artifacts(run_id, "config_space.json", dst_path=tmp_dir)

            with open(config_path, "r") as f:
                config_dict = json.load(f)

            # Reconstruct ConfigurationSpace
            config = ConfigurationSpace.from_dict(config_dict)
            config.generate_indices()

            # Extract original configuration from run metadata
            original_config = {
                'minimize': run_params.get('minimize', 'True').lower() == 'true',
                'acq_function': run_tags.get('acq_function', 'EI'),
                'batch_acquisition': run_tags.get('batch_acquisition', 'False').lower() == 'true',
                'acquisition_q': int(run_params.get('acquisition_q', '1')),
                'n_jobs': int(run_params.get('n_jobs', '1')),
                'model_checkpoint_freq': int(run_params.get('model_checkpoint_freq', '1')),
                'seed': int(run_tags.get('seed')) if run_tags.get('seed') and run_tags.get('seed') != '' else None,
                'experiment': run_info.experiment_id,
                'run_name': run_tags.get('mlflow.runName'),
                'tracking_uri': tracking_uri,
            }

            # Create optimizer instance WITHOUT initializing MLflow (we'll set it up manually)
            inst = cls.__new__(cls)  # Create instance without calling __init__

            # Set up the instance manually to avoid MLflow initialization
            inst.obj = obj
            inst.config = config
            inst.minimize = original_config['minimize']
            inst.sign_mul = -1 if original_config['minimize'] else 1
            inst.acq_function = original_config['acq_function']
            inst.acquisition_function_options = {}
            inst.batch_acquisition = original_config['batch_acquisition']
            inst.acquisition_q = original_config['acquisition_q']
            inst.n_jobs = original_config['n_jobs']
            inst.verbose = 1

            # Set up state attributes
            inst.model = None
            inst.train_confs = None
            inst.train_x = None
            inst.train_y = None
            inst.obj_eval_time = None
            inst.initial_params = None

            # Current iteration snapshot
            inst.curr_conf_inc = None
            inst.curr_conf_cand = None
            inst.curr_f_inc_obs = None
            inst.curr_f_inc_est = None
            inst.curr_fit_time = None
            inst.curr_acq_opt_time = None
            inst.curr_acq_val = None

            inst._pending_candidates = None
            inst._n_evals = 0

            # Set up MLflow config to reuse the existing run
            inst._model_checkpoint_freq = original_config['model_checkpoint_freq']
            inst._mlflow_initialized = True
            inst._total_iterations = 0
            inst._run = None
            inst._client = client
            inst._run_id = run_id
            inst._experiment_id = run_info.experiment_id
            inst._tracking_uri = tracking_uri or mlflow.get_tracking_uri()

            # Store MLflow config for potential reactivation
            inst._mlflow_config = {
                'tracking_uri': tracking_uri,
                'tracking_dir': None,
                'experiment': client.get_experiment(run_info.experiment_id).name,
                'run_name': original_config['run_name'],
                'seed': original_config['seed']
            }

            # Download and restore evaluation data
            eval_dir = os.path.join(tmp_dir, "evals")
            os.makedirs(eval_dir, exist_ok=True)

            # List and download all evaluation files
            try:
                eval_artifacts = client.list_artifacts(run_id, path="evals")
                eval_files = [item.path for item in eval_artifacts if item.path.endswith('.json')]
            except:
                # Fallback: try to find eval files recursively
                eval_files = []
                try:
                    all_artifacts = client.list_artifacts(run_id)
                    for artifact in all_artifacts:
                        if artifact.path.startswith('evals/') and artifact.path.endswith('.json'):
                            eval_files.append(artifact.path)
                except:
                    pass

            if not eval_files:
                raise RuntimeError("No evaluation data found in MLflow run")

            # Download and load evaluations
            evaluations = []
            for eval_file in sorted(eval_files):
                local_path = client.download_artifacts(run_id, eval_file, dst_path=tmp_dir)
                with open(local_path, 'r') as f:
                    evaluations.append(json.load(f))

            # Reconstruct training data
            inst.train_confs = [config.get_conf_from_array(np.array(eval_data["x"])) for eval_data in evaluations]
            inst.train_x = torch.tensor([eval_data["x"] for eval_data in evaluations]).double()
            inst.train_y = torch.tensor([eval_data["y"] for eval_data in evaluations]).double()
            inst.obj_eval_time = torch.tensor([eval_data["eval_time"] for eval_data in evaluations]).double()
            inst._n_evals = len(evaluations)

            # Find and load latest iteration snapshot to restore current state
            try:
                iter_artifacts = client.list_artifacts(run_id, path="iterations")
                iter_files = [item.path for item in iter_artifacts if item.path.endswith('.json')]

                if iter_files:
                    # Get the latest iteration file
                    latest_iter_file = sorted(iter_files)[-1]
                    local_path = client.download_artifacts(run_id, latest_iter_file, dst_path=tmp_dir)

                    with open(local_path, 'r') as f:
                        latest_snapshot = json.load(f)

                    # Restore current state from latest snapshot
                    if latest_snapshot.get('conf_inc'):
                        # Reconstruct configuration from the config dict
                        conf_dict = latest_snapshot['conf_inc']
                        inst.curr_conf_inc = CS.Configuration(config, conf_dict)

                    metrics = latest_snapshot.get('metrics', {})
                    inst.curr_f_inc_obs = metrics.get('f_inc_obs')
                    inst.curr_f_inc_est = metrics.get('f_inc_est')
                    inst.curr_fit_time = metrics.get('fit_time')
                    inst.curr_acq_opt_time = metrics.get('acq_opt_time')
                    inst.curr_acq_val = metrics.get('acq_val')

                    # Track total iterations completed
                    inst._total_iterations = latest_snapshot.get('iter', 0) + 1

            except Exception as e:
                print(f"Warning: Could not restore iteration state: {e}")

            # Load model checkpoint if available
            try:
                ckpt_artifacts = client.list_artifacts(run_id, path="checkpoints")
                ckpt_files = [item.path for item in ckpt_artifacts if item.path.endswith('.pth')]

                if ckpt_files:
                    if model_checkpoint == "latest":
                        ckpt_to_load = sorted(ckpt_files)[-1]
                    else:
                        ckpt_to_load = f"checkpoints/{model_checkpoint}"
                        if ckpt_to_load not in ckpt_files:
                            raise RuntimeError(f"Checkpoint '{model_checkpoint}' not found")

                    ckpt_path = client.download_artifacts(run_id, ckpt_to_load, dst_path=tmp_dir)

                    # Build model and load checkpoint
                    inst.model = inst._construct_model()
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    inst.model.load_state_dict(state_dict)

                    # Prepare warm-start parameters
                    inst.initial_params = OrderedDict()
                    for name, parameter in inst.model.named_parameters():
                        parameter.requires_grad_(False)
                        inst.initial_params[name] = parameter

            except Exception as e:
                print(f"Warning: Could not restore model checkpoint: {e}")
                inst.model = None
                inst.initial_params = None

            # Clear any pending candidates
            inst._pending_candidates = None

            return inst

        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ======================== PRIVATE METHODS ======================== #

    # ------------------------- MLflow helpers ------------------------- #
    def _initialize_mlflow(self):
        """Initialize MLflow tracking (called lazily on first run).

        Sets up MLflow tracking URI, experiment, and starts a new run. This method is
        called automatically on the first call to run() and handles directory creation
        for file-based tracking URIs.

        Note:
            Ends any existing active run to prevent UUID conflicts.
        """
        if self._mlflow_initialized:
            return

        # End any existing active run to prevent UUID conflicts
        if mlflow.active_run() is not None:
            mlflow.end_run()

        def _ensure_dir_exists(path: str):
            os.makedirs(path, exist_ok=True)
            return path

        tracking_uri = self._mlflow_config['tracking_uri']
        tracking_dir = self._mlflow_config['tracking_dir']

        if tracking_uri:
            if tracking_uri.startswith("file:"):
                local_path = tracking_uri[len("file:"):]
                _ensure_dir_exists(os.path.abspath(local_path))
            resolved_tracking_uri = tracking_uri
        else:
            base_dir = tracking_dir or "./mlruns"
            base_dir = os.path.abspath(base_dir)
            _ensure_dir_exists(base_dir)
            resolved_tracking_uri = f"file:{base_dir}"

        mlflow.set_tracking_uri(resolved_tracking_uri)

        # Set experiment
        exp_name = self._mlflow_config['experiment'] or "BayesOpt"
        mlflow.set_experiment(exp_name)

        # Start new run
        self._run = mlflow.start_run(run_name=self._mlflow_config['run_name'])
        self._client = MlflowClient()
        self._run_id = mlflow.active_run().info.run_id
        self._experiment_id = mlflow.active_run().info.experiment_id
        self._tracking_uri = resolved_tracking_uri

        # Log meta information
        mlflow.set_tags({
            "framework": "BayesOpt",
            "acq_function": self.acq_function,
            "batch_acquisition": str(self.batch_acquisition),
            "seed": str(self._mlflow_config['seed']) if self._mlflow_config['seed'] is not None else "",
        })

        mlflow.log_params({
            "minimize": self.minimize,
            "acquisition_q": self.acquisition_q,
            "n_jobs": self.n_jobs,
            "model_checkpoint_freq": self._model_checkpoint_freq,
        })

        # Write config space as an artifact
        self._log_config_space()
        self._mlflow_initialized = True

    def _ensure_mlflow_active(self):
        """Ensure MLflow run is active before logging.

        Checks if MLflow is initialized and if a run is currently active. If not initialized,
        calls _initialize_mlflow(). If initialized but no active run, reactivates the
        stored run using the run_id.

        This method is called before any MLflow logging operations to ensure proper state.
        """
        if not self._mlflow_initialized:
            self._initialize_mlflow()
        elif mlflow.active_run() is None:
            # Reactivate the run if it was ended
            mlflow.start_run(run_id=self._run_id)

    def _log_config_space(self):
        """Log the configuration space as an MLflow artifact.

        Serializes the configuration space to JSON and logs it as an artifact
        named 'config_space.json'.

        Note:
            Does not call _ensure_mlflow_active to avoid recursion during initialization.
        """
        # Don't call _ensure_mlflow_active here to avoid recursion
        mlflow.log_dict(self.config.to_serialized_dict(), artifact_file="config_space.json")

    def _log_eval(self, conf: Configuration, x: np.ndarray, y: float, eval_time: float):
        """Log a single evaluation as a JSON artifact.

        Records evaluation details as a JSON file without large tensors for efficient storage.

        Args:
            conf: Configuration that was evaluated.
            x: Numeric array representation of the configuration.
            y: Objective function value.
            eval_time: Time taken to evaluate the configuration.
        """
        self._ensure_mlflow_active()
        idx = self._n_evals
        payload = {
            "idx": idx,
            "conf": dict(conf),
            "x": np.asarray(x).tolist(),
            "y": float(y),
            "eval_time": float(eval_time),
        }
        mlflow.log_dict(payload, artifact_file=f"evals/eval_{idx:03d}.json")
        self._n_evals += 1

    def _log_iteration_snapshot(self, i: int):
        """
        Log per-iteration scalar metrics AND a small JSON snapshot of configs.

        - Metrics (logged with step=i): f_inc_obs, f_inc_est, fit_time, acq_opt_time, acq_val
        - Artifact  iterations/iter_<i>.json:
            {
            "iter": i,
            "conf_inc": {...},          # incumbent config as dict
            "conf_cand": [{...}, ...],  # candidate list as dicts
            "metrics": { ... }          # same scalars for convenient browsing
            }
        """
        self._ensure_mlflow_active()
        metrics = {
            "f_inc_obs": float(self.curr_f_inc_obs),
            "f_inc_est": float(self.curr_f_inc_est),
            "fit_time": float(self.curr_fit_time),
            "acq_opt_time": float(self.curr_acq_opt_time),
            "acq_val": float(self.curr_acq_val),
        }
        mlflow.log_metrics(metrics, step=i)

        # write compact per-iteration snapshot with configs
        snapshot = {
            "iter": int(i),
            "conf_inc": (dict(self.curr_conf_inc) if self.curr_conf_inc else None),
            "conf_cand": [dict(c) for c in (self.curr_conf_cand or [])],
            "metrics": metrics,
        }
        mlflow.log_dict(snapshot, artifact_file=f"iterations/iter_{i:03d}.json")

    def _save_and_log_model_state_iter(self, i: int, is_final: bool = False):
        """Checkpoint model weights based on frequency or if final iteration."""
        if self.model is None:
            return

        # Save checkpoint if: (1) at frequency interval, (2) final iteration, or (3) frequency is 1
        should_save = (
            (i + 1) % self._model_checkpoint_freq == 0 or  # At frequency interval
            is_final or                                      # Final iteration
            self._model_checkpoint_freq == 1                # Every iteration
        )

        if not should_save:
            return

        # Use iteration-specific filename to avoid overwrite conflicts
        fname = f"iter_{i:03d}_model_state.pth"

        # Make a stable staging dir for this run (once)
        if not hasattr(self, "_ckpt_stage_dir"):
            self._ckpt_stage_dir = os.path.join(
                tempfile.gettempdir(), f"bo_ckpt_{self._run_id}"
            )
            os.makedirs(self._ckpt_stage_dir, exist_ok=True)

        local_path = os.path.join(self._ckpt_stage_dir, fname)

        # Write the state dict to the deterministic path
        torch.save(self.model.state_dict(), local_path)

        # Log to MLflow. The artifact will be saved as checkpoints/<fname>
        self._ensure_mlflow_active()
        mlflow.log_artifact(local_path, artifact_path="checkpoints")

    def _get_iteration_data_from_artifacts(self) -> List[Dict[str, Any]]:
        """Retrieve iteration data from MLflow artifacts if available."""
        if not self._mlflow_initialized:
            return []

        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()

            # List iteration artifacts
            try:
                iter_artifacts = client.list_artifacts(self._run_id, path="iterations")
                iter_files = [item.path for item in iter_artifacts if item.path.endswith('.json')]
            except:
                return []

            if not iter_files:
                return []

            # Download and parse iteration files
            iteration_data = []
            with tempfile.TemporaryDirectory() as tmp_dir:
                for iter_file in sorted(iter_files):
                    try:
                        file_path = client.download_artifacts(self._run_id, iter_file, dst_path=tmp_dir)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if 'conf_inc' in data and 'f_inc_obs' in data:
                                iteration_data.append({
                                    'incumbent_config': data['conf_inc'],
                                    'observed_value': data['f_inc_obs']
                                })
                    except Exception:
                        continue

            return iteration_data

        except Exception:
            return []

    def _construct_iteration_data_from_state(self) -> List[Dict[str, Any]]:
        """Construct iteration data from current optimizer state when artifacts unavailable."""
        if not hasattr(self, 'train_confs') or not self.train_confs:
            return []

        # Reconstruct per-iteration incumbents by simulating the optimization process
        iteration_data = []

        # For each evaluation point, find the best configuration seen so far
        best_value = float('inf') if self.minimize else float('-inf')
        best_config = None

        for i, (config, value) in enumerate(zip(self.train_confs, self.train_y.numpy())):
            # train_y stores original objective values (not sign-multiplied)
            original_value = float(value)

            # Check if this is a new best
            is_improvement = (
                (self.minimize and original_value < best_value) or
                (not self.minimize and original_value > best_value)
            )

            if is_improvement or best_config is None:
                best_value = original_value
                best_config = config

            # Add iteration data
            iteration_data.append({
                'incumbent_config': dict(best_config) if hasattr(best_config, 'keys') else best_config,
                'observed_value': best_value
            })

        return iteration_data

    def _config_dict_to_array(self, config_dict: Dict[str, Any]) -> np.ndarray:
        """Convert configuration dictionary to array format for model input."""
        # Create Configuration object and get array
        if hasattr(self, 'config'):
            config_obj = Configuration(self.config, config_dict)
            return config_obj.get_array()
        else:
            # Fallback: convert based on known order (may not work in all cases)
            return np.array(list(config_dict.values()))

    # --------------------------- core methods --------------------------- #
    def _initialize(self, n_init: Optional[int]):
        """Initialize optimizer with random points or evaluate pending candidates.

        On first call, generates n_init random configurations and evaluates them.
        On subsequent calls, evaluates pending candidates from the last acquisition step.

        Args:
            n_init: Number of initial random points. Only used on first call.
        """
        if self.train_confs is None:
            if n_init is None:
                n_init = len(self.config.quant_index) + 1
            self.config.seed(np.random.randint(2e+4))
            self.train_confs = list(self.config.latinhypercube_sample(n_init))

            # evaluate initial design
            xs, ys, ts = [], [], []
            for conf in self.train_confs:
                x, y, t_eval = self._evaluate(conf)
                xs.append(x); ys.append(y); ts.append(t_eval)
                # log each evaluation to MLflow
                self._log_eval(conf, x, y, t_eval)

            self.train_x = torch.tensor(xs).double()
            self.train_y = torch.tensor(ys).double()
            self.obj_eval_time = torch.tensor(ts).double()
        else:
            # evaluate pending candidates from last acquisition
            if not self._pending_candidates:
                return
            xs, ys, ts = [], [], []
            for conf in self._pending_candidates:
                x, y, t_eval = self._evaluate(conf)
                xs.append(x); ys.append(y); ts.append(t_eval)
                self.train_confs.append(conf)
                self._log_eval(conf, x, y, t_eval)

            self.train_x = torch.cat([self.train_x, torch.tensor(xs).double().to(self.train_x)], dim=0)
            self.train_y = torch.cat([self.train_y, torch.tensor(ys).double().to(self.train_y)], dim=0)
            self.obj_eval_time = torch.cat([self.obj_eval_time, torch.tensor(ts).double().to(self.obj_eval_time)], dim=0)
            self._pending_candidates = None  # consumed

    def _evaluate(self, conf:Configuration, **kwargs) -> Tuple[np.ndarray, float, float]:
        """Evaluate the objective on a single configuration.

        Args:
            conf: A single configuration object.
            **kwargs: Additional keyword args passed to `self.obj`.

        Returns:
            tuple: (ndarray, float, float) where the first element is the
                numeric array from `conf.get_array()`, the second is the
                objective value, and the third is elapsed time.
        """
        start_time = time.time()
        y = self.obj(dict(conf),**kwargs)
        eval_time = time.time()-start_time
        return conf.get_array(),y,eval_time

    def _evaluate_confs(self, confs_list:List[Configuration],**kwargs) -> List[Tuple[np.ndarray, float, float]]:
        """Evaluate multiple configurations, optionally in parallel.

        Args:
            confs_list: List of configurations to evaulate the objective on.
            **kwargs: Additional keyword args passed to `self.obj`.

        Returns:
            List[tuple]: Each element is the return of `_evaluate`.
        """
        if self.n_jobs > 1 and len(confs_list) > 1:
            # enable parallel evaulations
            evaluations = joblib.Parallel(n_jobs=self.n_jobs,verbose=0)(
                joblib.delayed(self._evaluate)(conf,**kwargs) for conf in confs_list
            )
        else:
            # can add logging here
            evaluations = [None]*len(confs_list)
            for i,conf in enumerate(confs_list):
                evaluations[i] = self._evaluate(conf,**kwargs)
        
        return evaluations
    
    def _fit_model_and_find_inc(self, i: int) -> None:
        """Fit the Gaussian Process model and update the incumbent.

        Constructs a new GP model, optionally warm-starts it with previous parameters,
        fits it to the training data, and computes the current incumbent based on
        the predictive mean.

        Args:
            i: Current iteration number for logging and seeding.
        """
        # construct & (optionally) warm start
        self.model = self._construct_model()
        if self.initial_params is not None:
            self.model.initialize(**self.initial_params)

        t0 = time.time()
        _ = fit_model_scipy(model=self.model, num_restarts=5, n_jobs=self.n_jobs, rng_seed=i)
        self.curr_fit_time = time.time() - t0

        # freeze params and store for warm start
        self.initial_params = OrderedDict()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(False)
            self.initial_params[name] = parameter

        # compute incumbent from predictive mean
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter(action='ignore', category=gpytorch.utils.warnings.GPInputWarning)
            pred = self.model.predict(self.train_x)

        fidx = pred.argmax().item()  # argmax because model sees sign_mul * y
        self.curr_conf_inc = self.train_confs[fidx]
        self.curr_f_inc_obs = float(self.train_y[fidx].item())
        self.curr_f_inc_est = float(self.sign_mul * pred[fidx].item())

        # checkpoint model is now handled in main loop
    
    def _construct_model(self) -> GPR:
        """Construct a Gaussian Process Regression model.

        Creates a new GPR model instance with the current training data,
        applying sign multiplication for minimization/maximization.

        Returns:
            GPR: Configured GP model ready for fitting.
        """
        return GPR(
            train_x = self.train_x,
            train_y = self.sign_mul*self.train_y,
        ).double()
    
    def _acquisition(self, i: int) -> None:
        """Optimize acquisition function to propose next candidate(s).

        Creates the appropriate acquisition function based on the configured type,
        optimizes it to find the most promising configuration(s), and stores them
        as pending candidates for evaluation.

        Args:
            i: Current iteration number for logging.

        Raises:
            ValueError: If an unknown acquisition function is specified.
        """
        if self.acq_function == 'EI':
            best_f = -self.curr_f_inc_est if self.minimize else self.curr_f_inc_est
            acqobj = (qExpectedImprovement(self.model, best_f, sampler=SobolQMCNormalSampler(128, seed=0))
                      if self.batch_acquisition else ExpectedImprovement(self.model, best_f))
        elif self.acq_function == 'LCB':
            beta = torch.tensor(4.0, dtype=torch.double)
            acqobj = (qUpperConfidenceBound(self.model, beta, sampler=SobolQMCNormalSampler(128, seed=0))
                      if self.batch_acquisition else UpperConfidenceBound(self.model, beta))
        elif self.acq_function == 'KG':
            num_fantasies = 32
            acqobj = qKnowledgeGradient(self.model, sampler=SobolQMCNormalSampler(num_fantasies, seed=0),
                                        num_fantasies=num_fantasies)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acq_function}")

        t0 = time.time()
        new_x, max_acq = _optimize_botorch_acqf(
            acq_function=acqobj,
            d=self.train_x.shape[-1],
            q=1 if not self.batch_acquisition else self.acquisition_q,
            num_restarts=10 if self.acq_function == 'KG' else 20,
            n_jobs=self.n_jobs,
            raw_samples=128
        )
        self.curr_acq_opt_time = time.time() - t0

        xs = [row for row in (new_x if not torch.is_tensor(new_x) else list(new_x))]
        cand_confs = [self.config.get_conf_from_array(x.detach().cpu().numpy()) for x in xs]
        self.curr_conf_cand = cand_confs
        self._pending_candidates = cand_confs

        self.curr_acq_val = float((-max_acq.item()) if self.acq_function == 'LCB' else max_acq.item())

        # log scalar metrics only
        self._log_iteration_snapshot(i)