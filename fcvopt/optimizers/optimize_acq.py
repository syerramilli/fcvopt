import torch
import joblib
from botorch.acquisition import AcquisitionFunction, qKnowledgeGradient
from botorch.optim import optimize_acqf
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions
)
from typing import Tuple

def _optimize_botorch_acqf(
    acq_function: AcquisitionFunction,
    d: int, q:int,
    num_restarts: int = 10,
    n_jobs: int = 1,
    raw_samples:int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function to optimize a BoTorch acquisition function using 
    `botorch.optim.optimize_acqf`. Unlike the original function, this
    implementation allows for parallelization using joblib.


    Parameters
    ----------
    acqobj : AcquisitionFunction
        The acquisition function to optimize.
    d : int
        The number of dimensions.
    q : int
        The batch size.
    num_restarts : int, optional
        The number of restarts for the optimization. Default is 10.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The best location and value of the acquisition function.
    """
    
    # Generate initial conditions for the optimization
    bounds = torch.tensor([[0.0]*d, [1.0]*d], dtype=torch.double)
    options = {'maxiter': 500}

    # Generate initial conditions for the optimization
    if isinstance(acq_function, qKnowledgeGradient):
        initial_conditions = gen_one_shot_kg_initial_conditions(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples
        )
    else:
        initial_conditions = gen_batch_initial_conditions(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples
        )

    if n_jobs != 1:
        # define single‐restart worker
        def _worker(ic):
            # single restart, IC provided
            x_cand, acq_val = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=q,
                num_restarts=1,
                raw_samples=None,
                batch_initial_conditions=ic.unsqueeze(0),
                options=options,
            )
            # return shape [q, d], scalar
            return x_cand.squeeze(0), acq_val

        # 3) dispatch in parallel
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_worker)(initial_conditions[i]) for i in range(num_restarts)
        )
        # 4) pick the best
        Xs, Vs = zip(*results)
        Xs = torch.stack(Xs)                       # [R, q, d]
        Vs = torch.stack(Vs).view(-1)             # [R]
        best_idx = torch.argmax(Vs)
        new_x, max_acq = Xs[best_idx].unsqueeze(0), Vs[best_idx]
    else:
        # fallback to the default single‐call
        new_x, max_acq = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            batch_initial_conditions=initial_conditions
        )
    
    return new_x, max_acq