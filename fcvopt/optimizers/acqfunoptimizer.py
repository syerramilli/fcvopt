import numpy as np
import torch
from scipy.optimize import minimize,Bounds
from joblib import Parallel,delayed
from typing import Optional,Tuple,Dict,List
from ..optimizers.multistart import MultiStartOptimizer
from ..acquisition.acquisition import AcquisitionFunction
from ..configspace import ConfigurationSpace

#from ..util.samplers import lh_sampler

class AcqFunOptimizer(MultiStartOptimizer):
    def __init__(
        self,
        acq_fun:AcquisitionFunction,
        ndim:int,
        **kwargs
    ) -> None:
        self.acq_fun = acq_fun
        self.obj_sign = -1 if self.acq_fun.maximize else 1
        super().__init__(
            obj = self.scipy_obj,
            lb = np.zeros(ndim),
            ub = np.ones(ndim),
            jac = True,
            **kwargs
        )
    
    def scipy_obj(self,x:np.ndarray) -> Tuple[float,np.ndarray]:
        X = (
            torch.from_numpy(x)
            .double()
            .view(1,-1)
            .contiguous()
            .requires_grad_(True)
        )
        loss =  self.obj_sign*self.acq_fun(X)
        # X is a non-leaf variable; need to manually extract
        # the gradient. Flip side is there is no need to
        # zero the gradients - don't accumulate
        grad_f = torch.autograd.grad(loss,X)[0].contiguous().view(-1).double().numpy()
        fval = loss.item()
        return fval,grad_f
