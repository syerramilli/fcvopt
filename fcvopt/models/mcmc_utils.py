import warnings
import torch
import numpy as np
import gpytorch
import pyro
import gpytorch.settings as gptsettings
from gpytorch.utils.errors import NanError,NotPSDError
from pyro.infer.mcmc import MCMC,NUTS
from pyro.infer.autoguide import init_to_value

from . import HGP
from ._pyro_models import pyro_gp,pyro_hgp

from typing import Optional, Dict, Tuple
from collections import OrderedDict
from copy import deepcopy

def get_samples(samples,num_samples=None, group_by_chain=False):
    """
    Get samples from the MCMC run

    :param int num_samples: Number of samples to return. If `None`, all the samples
        from an MCMC chain are returned in their original ordering.
    :param bool group_by_chain: Whether to preserve the chain dimension. If True,
        all samples will have num_chains as the size of their leading dimension.
    :return: dictionary of samples keyed by site name.
    """
    if num_samples is not None:
        batch_dim = 0
        sample_tensor = list(samples.values())[0]
        batch_size, device = sample_tensor.shape[batch_dim], sample_tensor.device
        idxs = torch.linspace(0,batch_size-1,num_samples,dtype=torch.long,device=device).flip(0)
        samples = {k: v.index_select(batch_dim, idxs) for k, v in samples.items()}
    return samples

def run_hmc_seq(
    model:gpytorch.models.ExactGP,
    num_samples:int=500,
    warmup_steps:int=500,
    num_model_samples:int=50,
    disable_progbar:bool=True,
    init_params:Dict=None,
    max_tree_depth:int=5,
    jit_compile:bool=False
):
    if init_params is None:
        init_values={}
        for name,module,prior,closure,_ in model.named_priors():
            init_values[name] = prior.expand(closure(module).shape).sample()

        init_params = {
            'step_size':0.1,
            'inverse_mass_matrix':None,
            'init_values':init_values,
        }
    
    if isinstance(model,HGP):
        kwargs = {
            'x_aug':model.train_inputs,
            'y':model.train_targets,
            'jitter':model.likelihood.noise_covar.raw_noise_constraint.lower_bound.item()
        }

        pyro_model = pyro_hgp
    else:
        kwargs = {
            'x':model.train_inputs[0],
            'y':model.train_targets,
            'jitter':model.likelihood.noise_covar.raw_noise_constraint.lower_bound.item()
        }

        pyro_model = pyro_gp
    
    hmc_kernel = NUTS(
        pyro_model,
        step_size=init_params['step_size'],
        adapt_step_size=True,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_value(values=init_params['init_values']),
        full_mass=False,
        jit_compile=jit_compile,
    )
    if init_params['inverse_mass_matrix'] is not None:
        setattr(
            hmc_kernel.mass_matrix_adapter,
            'inverse_mass_matrix',
            init_params['inverse_mass_matrix']
        )

    mcmc_run = MCMC(
        kernel=hmc_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        disable_progbar=disable_progbar,
        num_chains=1,
    )
    
    # run mcmc
    mcmc_run.run(**kwargs)
    samples = {k:v for k,v in get_samples(deepcopy(mcmc_run.get_samples()),num_model_samples).items()}

    # load_samples and remove gradients
    model.pyro_load_from_samples(samples)
    # removing gradients on the model hyperparameters
    for _,parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
    
    # return last parameter state
    last_params = {
        'step_size':hmc_kernel.step_size,
        'inverse_mass_matrix':hmc_kernel.mass_matrix_adapter.inverse_mass_matrix,
        'init_values':  {k:v[-1,...] for k,v in deepcopy(mcmc_run.get_samples()).items()}
    }

    return mcmc_run,last_params