import re
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS,HMC,MCMC
from .fit_model import fit_model_unconstrained
from typing import Optional, Dict, Tuple

def pyro_model(
    model:gpytorch.models.ExactGP,
    mll:gpytorch.mlls.ExactMarginalLogLikelihood
):
    model.pyro_sample_from_prior()
    output = model(model.train_inputs[0])
    loss = mll.pyro_factor(output,model.train_targets)
    return model.train_targets

def mcmc_run(
    model:gpytorch.models.ExactGP,
    step_size:Optional[float]=None,
    num_samples:int=100,
    warmup_steps:int=100,
    num_model_samples:int=30,
    disable_progbar:bool=True
) -> Tuple[Dict,float]:
    adapt_step_size=False
    
    if step_size is None:
        adapt_step_size = True
        step_size = 1
    
    # initialize HMC with a local model of the posterior
    with gpytorch.settings.fast_computations(log_prob=False):
        _ = fit_model_unconstrained(model,num_restarts=0)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)
    nuts_kernel = NUTS(pyro_model,step_size=step_size,
                      adapt_step_size=adapt_step_size)
    mcmc = MCMC(
        kernel=nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        disable_progbar=disable_progbar
    )

    # run mcmc
    mcmc.run(model=model,mll=mll)

    # load_samples and remove gradients
    model.pyro_load_from_samples(mcmc.get_samples(num_model_samples))
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # # initial params to the next cycle
    # initial_params = transform_param_dict({k:v.squeeze(0) for k,v in mcmc.get_samples(1).items()})
    # step_size = nuts_kernel.step_size

    # del mcmc,nuts_kernel

    return step_size