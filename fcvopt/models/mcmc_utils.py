import re
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS,HMC,MCMC
from .fit_model import fit_model_unconstrained
from typing import Optional, Dict, Tuple
from collections import OrderedDict

def pyro_model(
    model:gpytorch.models.ExactGP,
    mll:gpytorch.mlls.ExactMarginalLogLikelihood
):
    model.pyro_sample_from_prior()
    output = model(*model.train_inputs)
    loss = mll.pyro_factor(output,model.train_targets)
    return model.train_targets

def mcmc_run(
    model:gpytorch.models.ExactGP,
    initial_params:Optional[Dict]=None,
    step_size:float=1.0,
    adapt_step_size:bool=True,
    num_samples:int=100,
    warmup_steps:int=100,
    num_model_samples:int=30,
    disable_progbar:bool=True
) -> Tuple[Dict,float]:
    
    if initial_params is not None:
        model.initialize(**initial_params)
    # else:
    #     # initialize HMC with a local model of the posterior
    #     with gpytorch.settings.fast_computations(log_prob=False):
    #         _ = fit_model_unconstrained(model,num_restarts=0)
    #         print(list(model.named_parameters()))
    
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
    initial_params = OrderedDict()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            initial_params[name]=parameter[0,...]

    return initial_params