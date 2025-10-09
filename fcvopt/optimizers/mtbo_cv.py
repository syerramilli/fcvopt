import numpy as np
import time
import torch
from botorch.acquisition import ExpectedImprovement

from .optimize_acq import _optimize_botorch_acqf
from .fcvopt import FCVOpt
from ..models.gpregression import GPR
from ..models.multitaskgp import MultitaskGPModel, MultiTaskGPConstantCorrModel
from ..fit.mll_scipy import fit_model_scipy

from ..configspace import ConfigurationSpace
from typing import Callable, Optional

class SingleTaskExpectedImprovement(ExpectedImprovement):
    def __init__(self,model:MultitaskGPModel,task_idx:int,best_task:float,**kwargs):
        super().__init__(model=model,best_f=best_task,**kwargs)
        self.register_buffer('task_idx',torch.tensor([[task_idx]]).long()) 
    
    def _mean_and_sigma(self,X:torch.Tensor):
        mean,sigma = self.model.predict(X,self.task_idx.repeat(X.shape[-2],1),return_std=True)
        mean = mean.squeeze(-1)
        sigma = sigma.squeeze(-1)
        return mean,sigma


class MTBOCVOpt(FCVOpt):
    def __init__(
        self,
        obj:Callable,
        n_folds:int,
        config:ConfigurationSpace,
        fold_initialization:str='random',
        minimize:bool=True,
        fold_selection_criterion:str='single-task-ei',
        constant_task_corr:bool=False,
        **kwargs
    ):
        super().__init__(
            obj=obj,config=config,
            n_folds=n_folds,n_repeats=1,
            fold_selection_criterion=fold_selection_criterion,
            fold_initialization=fold_initialization,
            minimize=minimize,acq_function=None,
            **kwargs
        )
        self.constant_task_corr = constant_task_corr
    
    def _construct_model(self):
        if self.constant_task_corr:
            return MultiTaskGPConstantCorrModel(
                train_x = (self.train_x,self.train_folds),
                train_y = self.sign_mul*self.train_y,
                num_tasks=self.n_folds
            ).double()

        return MultitaskGPModel(
            train_x = (self.train_x,self.train_folds),
            train_y = self.sign_mul*self.train_y,
            num_tasks=self.n_folds
        ).double()
    
    def _create_multitask_average_model(self):
        """Create GP model for the average of multi-task predictions.

        Returns:
            GPR: GP model fitted to average task predictions.
        """
        # obtain predictions for the average of tasks
        with torch.no_grad():
            new_train_y = self.model.predict(self.train_x)

        # fit a new surrogate model for the average of tasks
        new_model = GPR(self.train_x, new_train_y)
        _ = fit_model_scipy(new_model, num_restarts=5)
        _ = new_model.eval()

        return new_model

    def _select_next_candidates(self, i: int):
        """Select next candidate configurations using multi-task EI on average tasks.

        Args:
            i: Current iteration number for logging.

        Returns:
            List[Configuration]: Selected candidate configurations.
        """
        del i  # Silence unused parameter warning

        # Create average task model
        avg_model = self._create_multitask_average_model()

        # Get average predictions
        with torch.no_grad():
            avg_train_y = self.model.predict(self.train_x)

        # optimize EI on the average of tasks
        acqobj = ExpectedImprovement(avg_model, best_f=avg_train_y.max())

        t0 = time.time()
        new_x, max_acq = _optimize_botorch_acqf(
            acq_function=acqobj,
            d=self.train_x.shape[-1],
            q=1,
            num_restarts=20,
            n_jobs=self.n_jobs,
            raw_samples=128
        )
        self.curr_acq_opt_time = time.time() - t0

        # Convert to configurations
        cand_confs = [self.config.get_conf_from_array(x.detach().cpu().numpy()) for x in new_x]
        self.curr_acq_val = float(max_acq.item())

        return cand_confs

    def _select_fold_indices(self, cand_confs):
        """Select fold indices using multi-task specific criteria.

        Args:
            cand_confs: List of candidate configurations.

        Returns:
            List[int]: Selected fold indices for each candidate.
        """
        fold_idxs = np.arange(self.n_folds)

        if self.fold_selection_criterion == 'random':
            selected_folds = np.random.default_rng(0).choice(
                fold_idxs, size=len(cand_confs), replace=True
            ).tolist()

        elif self.fold_selection_criterion == 'single-task-ei':
            selected_folds = []

            # Convert configurations back to tensor format for acquisition evaluation
            new_x = torch.stack([torch.from_numpy(conf.get_array()).double() for conf in cand_confs])

            for x in new_x:
                # shuffling to prevent ties among folds
                fold_idxs_shuffled = fold_idxs.copy()
                np.random.default_rng(0).shuffle(fold_idxs_shuffled)

                fold_metrics = []
                for fold_idx in fold_idxs_shuffled:
                    with torch.no_grad():
                        y_fold_idx = self.model.predict(
                            self.train_x, i=torch.tensor([[fold_idx]]).long().repeat(self.train_x.shape[0], 1)
                        )

                    acqobj_fold = SingleTaskExpectedImprovement(self.model, fold_idx, y_fold_idx.max())
                    with torch.no_grad():
                        fold_metrics.append(
                            acqobj_fold(x.unsqueeze(0)).item()
                        )

                selected_folds.append(fold_idxs_shuffled[np.argmax(fold_metrics)])
        else:
            raise ValueError(f"Unknown fold selection criterion: {self.fold_selection_criterion}")

        return selected_folds

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
        constant_task_corr: bool = False,
        fold_selection_criterion: str = 'single-task-ei',
        **kwargs
    ) -> "MTBOCVOpt":
        """Restore an MTBOCVOpt instance from MLflow with multi-task parameters.

        This extends the FCVOpt restoration to handle MTBOCVOpt-specific parameters.

        Args:
            obj: Objective function to use for the restored optimizer.
            run_id: MLflow run ID to restore from.
            experiment_name: MLflow experiment name (used with run_name).
            run_name: MLflow run name (used with experiment_name).
            tracking_uri: MLflow tracking URI.
            tracking_dir: MLflow tracking directory (alternative to tracking_uri).
            model_checkpoint: Model checkpoint to load.
            constant_task_corr: Whether to use constant task correlation model.
            fold_selection_criterion: Fold selection strategy for multi-task BO.
            **kwargs: Additional keyword arguments for FCVOpt.

        Returns:
            MTBOCVOpt: Restored optimizer instance.
        """
        # Use FCVOpt restoration as base, but override fold_selection_criterion
        fcv_instance = super().restore_from_mlflow(
            obj=obj,
            run_id=run_id,
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tracking_dir=tracking_dir,
            model_checkpoint=model_checkpoint,
            fold_selection_criterion=fold_selection_criterion,
            **kwargs
        )

        # Create MTBOCVOpt instance with the same state
        mtbo_instance = cls(
            obj=obj,
            config=fcv_instance.config,
            n_folds=fcv_instance.n_folds,
            fold_initialization=fcv_instance.fold_initialization,
            minimize=fcv_instance.minimize,
            fold_selection_criterion=fold_selection_criterion,
            constant_task_corr=constant_task_corr,
            **kwargs
        )

        # Copy all restored state from FCVOpt instance
        for attr in ['train_confs', 'train_x', 'train_y', 'train_folds', 'obj_eval_time', '_n_evals',
                     'curr_conf_inc', 'curr_f_inc_obs', 'curr_f_inc_est',
                     'curr_fit_time', 'curr_acq_opt_time', 'curr_acq_val',
                     '_total_iterations', '_run_id', '_experiment_id', '_tracking_uri',
                     'model', 'initial_params', '_mlflow_initialized', '_client', 'folds_cand']:
            if hasattr(fcv_instance, attr):
                setattr(mtbo_instance, attr, getattr(fcv_instance, attr))

        return mtbo_instance