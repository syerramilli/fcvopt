import time
import warnings
from typing import Callable, Optional

import torch
from botorch.acquisition import qNegIntegratedPosteriorVariance
from botorch.sampling import SobolQMCNormalSampler

from ..configspace import ConfigurationSpace
from .bayes_opt import BayesOpt
from .optimize_acq import _optimize_botorch_acqf


class ActiveLearning(BayesOpt):
    """Active learning via integrated posterior variance reduction.

    Selects configurations that most reduce GP uncertainty integrated over a
    reference set ``X_ref`` (``qNegIntegratedPosteriorVariance``). Unlike
    :class:`BayesOpt`, there is no optimum to find—the goal is to learn the
    function shape across the domain efficiently.

    All MLflow tracking, model checkpointing, warm-starting, parallel
    evaluation, and continuation/restore logic are inherited from
    :class:`BayesOpt`.

    Args:
        obj: Objective function mapping a configuration dict to a scalar.
        config: Hyperparameter search space.
        X_ref: Reference tensor of shape ``(n_ref, d)`` used to compute
            integrated posterior variance. Required.
        n_jobs: Number of parallel jobs. Defaults to 1.
        verbose: Verbosity level (0/1/2). Defaults to 1.
        seed: Random seed. Defaults to None.
        tracking_uri: MLflow tracking URI. Defaults to None.
        tracking_dir: Directory for MLflow tracking. Defaults to None.
        experiment: MLflow experiment name. Defaults to ``"ActiveLearning"``.
        run_name: MLflow run name. Defaults to a timestamp string.
        model_checkpoint_freq: Save GP checkpoint every N iterations.
            Defaults to 1.

    Examples:
        >>> al = ActiveLearning(obj=measure, config=cs, X_ref=X_ref)
        >>> al.run(n_iter=20, n_init=5)
    """

    def __init__(
        self,
        obj: Callable,
        config: ConfigurationSpace,
        X_ref: torch.Tensor,
        n_jobs: int = 1,
        verbose: int = 1,
        seed: Optional[int] = None,
        tracking_uri: Optional[str] = None,
        tracking_dir: Optional[str] = None,
        experiment: Optional[str] = None,
        run_name: Optional[str] = None,
        model_checkpoint_freq: int = 1,
    ):
        super().__init__(
            obj=obj,
            config=config,
            minimize=False,        # sign_mul=1; GP trained on raw y values
            acq_function='EI',     # placeholder; overridden by _create_acquisition_function
            verbose=verbose,
            n_jobs=n_jobs,
            seed=seed,
            tracking_uri=tracking_uri,
            tracking_dir=tracking_dir,
            experiment=experiment,
            run_name=run_name,
            model_checkpoint_freq=model_checkpoint_freq,
        )
        self.X_ref = X_ref
        # Overwrite placeholder so MLflow logs the correct acquisition name
        self.acq_function = 'NIPV'

    def _create_acquisition_function(self):
        """Return a qNegIntegratedPosteriorVariance acquisition over X_ref."""
        return qNegIntegratedPosteriorVariance(
            self.model,
            mc_points=self.X_ref,
            # Dummy sampler — NIPV does not depend on y samples
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=0),
        )

    def _select_next_candidates(self, i: int):
        """Optimize NIPV to select the next configuration.

        Overrides the parent to (a) use the NIPV acquisition and (b) negate
        ``max_acq`` so that ``curr_acq_val`` reflects actual integrated
        posterior variance (positive) rather than the negated value returned
        by BoTorch.
        """
        del i
        acqobj = self._create_acquisition_function()

        t0 = time.time()
        new_x, max_acq = _optimize_botorch_acqf(
            acq_function=acqobj,
            d=self.train_x.shape[-1],
            q=1,
            num_restarts=20,
            n_jobs=self.n_jobs,
            raw_samples=128,
        )
        self.curr_acq_opt_time = time.time() - t0

        xs = list(new_x) if torch.is_tensor(new_x) else new_x
        cand_confs = [self.config.get_conf_from_array(x.detach().cpu().numpy()) for x in xs]
        # NIPV maximises negative variance; negate so curr_acq_val = actual variance
        self.curr_acq_val = float(-max_acq.item())
        return cand_confs

    def _print_summary(self, status_msg: str) -> None:
        """Print active-learning summary (variance-focused)."""
        print(f'\nNumber of candidates evaluated.....: {len(self.train_confs)}')
        print(f'Integrated posterior variance.......: {self.curr_acq_val:.6g}')
        print(f'\n Last candidate {status_msg}:\n', self.curr_conf_inc)
