import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval

from ..priors import BetaPrior

class HammingKernel(Kernel):
    has_lengthscale=False
    def __init__(self,**kwargs):
        super().__init__(has_lengthscale=False,**kwargs)
        
        # register correlation parameter
        # TODO: add support for multiple indicies
        self.register_parameter(
            name='raw_rho',
            parameter=torch.nn.Parameter(torch.randn(*self.batch_shape,1,1))
        )

        self.register_constraint('raw_rho',Interval(0.,1.))
        self.register_prior(
            'rho_prior',
            BetaPrior(1.,5.),
            param_or_closure=lambda: self.rho,
            setting_closure= lambda v: self._set_rho(v)
        )
    
    @property
    def rho(self):
        return self.raw_rho_constraint.transform(self.raw_rho)
    
    def _set_rho(self,value):
        raw_value = (
            self.raw_rho_constraint.
            inverse_transform(value.to(self.raw_rho))
        )
        self.initialize(raw_rho=raw_value)
    
    def forward(
        self,
        x1,x2,
        diag=False,
        last_dim_is_batch=False
    ):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        res = None
        if diag:
            # Special case the diagonal because we can return all ones most of the time.
            if x1_eq_x2:
                res = torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                res = 1.-torch.norm(x1,x2,p=0,dim=-1)
        
        else:
            dist = torch.cdist(x1,x2,p=0.)
            res = 1. - dist.mul(1.-self.rho)
        
        return res


class ZeroOneKernel(Kernel):
    has_lengthscale=False
    def __init__(self,**kwargs):
        super().__init__(has_lengthscale=False,**kwargs)
    
    def forward(
        self,
        x1,x2,
        diag=False,
        last_dim_is_batch=False
    ):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        res = None
        if diag:
            # Special case the diagonal because we can return all ones most of the time.
            if x1_eq_x2:
                res = torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                res = 1.-torch.norm(x1,x2,p=0,dim=-1)
        
        else:
            res = 1. - torch.cdist(x1,x2,p=0.)
        
        return res