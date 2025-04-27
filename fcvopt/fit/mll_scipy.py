import torch
import numpy as np
from gpytorch import settings as gptsettings
from scipy.optimize import minimize, OptimizeResult
from collections import OrderedDict
from functools import reduce
from typing import Dict, List, Tuple, Optional, Union
from copy import deepcopy
from joblib import Parallel, delayed

def marginal_log_likelihood(model,add_prior:bool):
    output = model(*model.train_inputs)
    out = model.likelihood(output).log_prob(model.train_targets)
    if add_prior:
        # add priors
        for _, module, prior, closure, _ in model.named_priors():
            out.add_(prior.log_prob(closure(module)).sum())

    # loss terms
    for added_loss_term in model.added_loss_terms():
        out.add_(added_loss_term.loss().sum())
        
    return out

class MLLObjective:
    """
    Wrapper for the maximum (log-)likelihood or log-posterior objective
    of a GP model, suitable for SciPy optimization.

    Parameters
    ----------
    model : models.GPR
        A Gaussian process model whose hyperparameters will be optimized.
    add_prior : bool, optional
        If True, include the log-prior contributions in the objective.
        Default is True.

    Attributes
    ----------
    model : same as input
        The GP model instance.
    add_prior : bool
        Whether to include priors in log-likelihood.
    param_shapes : OrderedDict
        Shapes of each parameter tensor, to pack/unpack vectors.
    """
    def __init__(self, model, add_prior: bool = True):
        self.model = model
        self.add_prior = add_prior

        parameters = OrderedDict([
            (n,p) for n,p in self.model.named_parameters() if p.requires_grad
        ])
        self.param_shapes = OrderedDict()
        
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                if len(parameters[n].size()) > 0:
                    self.param_shapes[n] = parameters[n].size()
                else:
                    self.param_shapes[n] = torch.Size([1])

    def pack_parameters(self) -> np.ndarray:
        """
        Pack model parameters into a flat numpy array.

        Returns
        -------
        x : np.ndarray
            Flattened parameter vector.
        """
        parameters = OrderedDict([
            (n,p) for n,p in self.model.named_parameters() if p.requires_grad
        ])
        
        return np.concatenate([parameters[n].data.numpy().ravel() for n in parameters])

    def unpack_parameters(self, x: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Unpack a flat array to a state_dict-compatible mapping.

        Parameters
        ----------
        x : np.ndarray
            Flat parameter vector of length equal to total number of trainable parameters.

        Returns
        -------
        param_dict : Dict[str, torch.Tensor]
            Dictionary mapping parameter names to reshaped tensors.
        """
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters
    
    def pack_grads(self) -> np.ndarray:
        """
        Flatten the gradients of all trainable parameters into a 1D numpy array.

        Returns
        -------
        grad_vector : ndarray
            Gradient vector matching the flattened parameter vector.
        """
        grads = []
        for name,p in self.model.named_parameters():
            if p.requires_grad:
                grad = p.grad.data.numpy()
                grads.append(grad.ravel())
        return np.concatenate(grads).astype(np.float64)

    def fun(self, x:np.ndarray,return_grad=True) -> Union[float,Tuple[float,np.ndarray]]:
        """
        Compute the negative log-likelihood (plus priors) and its gradient.

        Parameters
        ----------
        x : ndarray
            Flattened hyperparameter vector.
        return_grad : bool, optional
            If True, also return the gradient. Default is True.

        Returns
        -------
        obj_val : float
            Value of the negative log-likelihood (or log-posterior).
        grad_vector : ndarray
            Gradient of the objective w.r.t. parameters (only if return_grad).
        """
        # unpack x and load into module 
        state_dict = self.unpack_parameters(x)
        old_dict = self.model.state_dict()
        old_dict.update(state_dict)
        self.model.load_state_dict(old_dict)
        
        # zero the gradient
        self.model.zero_grad()
        obj = -marginal_log_likelihood(self.model, self.add_prior) # negative sign to minimize
        
        if return_grad:
            # backprop the objective
            obj.backward()
            
            return obj.item(),self.pack_grads()
        
        return obj.item()

def fit_model_scipy(
    model,
    add_prior: bool = True,
    num_restarts: int = 5,
    options: Dict[str, Union[int, float]] = {},
    n_jobs: int = 1
) -> Tuple[List[OptimizeResult], float]:
    """
    Optimize GP hyperparameters via SciPy's L-BFGS-B, with parallel restarts.

    Parameters
    ----------
    model : gpytorch.models.ExactGP or models.GPR
        Gaussian process model to optimize.
    add_prior : bool, default=True
        Include parameter priors in the log-posterior.
    num_restarts : int, default=5
        Number of random-restart local optimizations.
    options : dict, optional
        Options passed to SciPy's L-BFGS-B solver (e.g. 'maxiter', 'gtol').
    n_jobs : int, default=1
        Number of parallel jobs for restarts (-1 means use all cores).

    Returns
    -------
    results : list of OptimizeResult
        SciPy optimization result for each restart.
    best_fun : float
        Lowest objective value found across all restarts.
    """
    # Validate n_jobs
    if not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
        raise ValueError(f"n_jobs must be -1 or a positive integer; got {n_jobs!r}")

    # default options
    defaults = {
        'ftol':1e-6,'gtol':1e-5,'maxfun':500,'maxiter':200
    }
    if len(options) > 0:
        for key in options.keys():
            if key not in defaults.keys():
                raise RuntimeError('Unknown option %s!'%key)
            defaults[key] = options[key]

    def _restart_worker(i: int):
        # Deep copy model per process
        model_i = deepcopy(model)
        # Initialize parameters
        if i > 0:
            model_i.reset_parameters()

        # wrap objective
        lik_i = MLLObjective(model_i, add_prior)
        # optimize
        res, best_state = None, None
        try:
            with gptsettings.fast_computations(log_prob=False):
                res = minimize(
                    fun = lik_i.fun,
                    x0 = lik_i.pack_parameters(),
                    args=(True),
                    method = 'L-BFGS-B',
                    bounds=None,
                    jac=True,
                    options=defaults
                )
            if isinstance(res, OptimizeResult):
                best_state = lik_i.unpack_parameters(res.x)
                
        except Exception as e:
            res = e

        return res, best_state
        

    # Run all restarts via joblib
    outputs = Parallel(n_jobs=n_jobs)(
        (delayed(_restart_worker)(i) for i in range(num_restarts + 1))
    )

    # Collect results
    results = []
    best_fun = float('inf')
    best_state = None
    for res, state in outputs:
        results.append(res)
        if hasattr(res, 'fun') and res.fun < best_fun:
            best_fun = res.fun
            best_state = state

    # Load best into original model
    current_state_dict = deepcopy(model.state_dict())
    if best_state is not None:
        current_state_dict.update(best_state)
        model.load_state_dict(current_state_dict)

    return results, best_fun