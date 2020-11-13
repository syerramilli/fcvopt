import torch
import numpy as np
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from collections import OrderedDict
from functools import reduce
from typing import Any,Dict,List
from copy import deepcopy
from scipy.optimize import minimize,OptimizeResult

class MLLObjective:
    """Log-likelihood objective function, wrapped to be called by scipy.optimize."""
    def __init__(self,model):
        self.mll = ExactMarginalLogLikelihood(model.likelihood,model)

        parameters = OrderedDict([
            (n,p) for n,p in self.mll.model.named_parameters() if p.requires_grad
        ])
        self.param_shapes = OrderedDict()
        for n,p in self.mll.model.named_parameters():
            if p.requires_grad:
                if len(parameters[n].size()) > 0:
                    self.param_shapes[n] = parameters[n].size()
                else:
                    self.param_shapes[n] = torch.Size([1])
    
    def pack_parameters(self):
        '''
        Returns the current hyperparameters in vector form for the scipy optimizer
        '''
        parameters = OrderedDict([
            (n,p) for n,p in self.mll.model.named_parameters() if p.requires_grad
        ])
        
        return np.concatenate([parameters[n].data.numpy().ravel() for n in parameters])
    
    def sample_from_prior(self):
        '''
        Samples hyperparameters and modifies them in place
        '''

        # sample the remaining hyperparameters from their respective priors
        # Note: samples in place
        for _,prior,closure,setting_closure in self.mll.model.named_priors():
            num_samples = (1,) if len(prior.shape()) > 0 else closure().shape
            setting_closure(prior.sample(num_samples))
    
    def unpack_parameters(self, x):
        """optimize.minimize will supply 1D array, chop it up for each parameter."""
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

    def pack_grads(self):
        """pack all the gradients from the parameters in the module into a
        numpy array."""
        grads = []
        for name,p in self.mll.model.named_parameters():
            if p.requires_grad:
                grad = p.grad.data.numpy()
                grads.append(grad.ravel())
        return np.concatenate(grads).astype(np.float64)

    def fun(self, x,return_grad=True):
        # unpack x and load into module 
        state_dict = self.unpack_parameters(x)
        old_dict = self.mll.model.state_dict()
        old_dict.update(state_dict)
        self.mll.model.load_state_dict(old_dict)
        
        # zero the gradient
        self.mll.zero_grad()
        # use it to calculate the objective
        output = self.mll.model(self.mll.model.train_inputs[0])
        obj = -self.mll(output,self.mll.model.train_targets) # negative sign to minimize
        
        if return_grad:
            # backprop the objective
            obj.backward()
            return obj.item(),self.pack_grads()
        
        return obj.item()

def fit_model_unconstrained(
    model:gpytorch.models.ExactGP,
    num_restarts:int=5,
    options:Dict={}
    ) -> List[OptimizeResult]:

    likobj = MLLObjective(model)
    current_state_dict = deepcopy(likobj.mll.state_dict())

    f_inc = np.inf
    # Output - Contains either optimize result objects or exceptions
    out = []
    
    for i in range(num_restarts+1):
        try:
            res = minimize(
                fun = likobj.fun,
                x0 = likobj.pack_parameters(),
                method = 'L-BFGS-B',
                jac=True,
                bounds=None,
                options=options
            )
            out.append(res)
            
            if res.fun < f_inc:
                current_state_dict = deepcopy(likobj.mll.state_dict())
                f_inc = res.fun
        except Exception as e:
            out.append(e)
        
        likobj.mll.load_state_dict(current_state_dict)
        if i < num_restarts:
            # replaces prior in place
            likobj.sample_from_prior()

    # load final dictionary
    likobj.mll.load_state_dict(current_state_dict)
    return out,f_inc