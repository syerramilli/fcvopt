import torch
import math
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive,GreaterThan
from linear_operator.operators import (
    RootLinearOperator,
    PsdSumLinearOperator,
    InterpolatedLinearOperator
)
#from gpytorch.utils.broadcasting import _mul_broadcast_shape

class MultiTaskKernel(Kernel):
    '''
    Multitask kernel used by Swersky et.al, 2014
    '''
    def __init__(self,num_tasks:int,**kwargs):
        super().__init__(**kwargs)

        num_params = int((num_tasks*(num_tasks+1))/2)
        
        self.register_parameter(
            name='raw_chol_vec',
            parameter=torch.nn.Parameter(torch.randn(*self.batch_shape,num_params))
        )
        self.register_constraint(
            'raw_chol_vec',
            GreaterThan(1e-6,transform=torch.exp,inv_transform=torch.log)
        )
    
    @property
    def chol_vec(self):
        return self.raw_chol_vec_constraint.transform(self.raw_chol_vec)
    
    def _set_chol_vec(self,value):
        raw_value = (
            self.raw_chol_vec_constraint.
            inverse_transform(value.to(self.raw_chol_vec))
        )
        self.initialize(raw_chol_vec=raw_value)
    
    @property
    def chol_factor(self):
        # recreates fill_triangular from tensorflow
        # The key trick is to create an upper triangular matrix by concatenating 
        # `x` and a tail of itself, then reshaping.
        
        # first find n by solving m=n(n+1)/2
        m = self.raw_chol_vec.shape[-1]
        n = int(math.sqrt(0.25 + 2. * m) - 0.5)

        chol_vec= self.chol_vec
        
        tmp = torch.cat([
            chol_vec[...,n:],chol_vec.flip(dims=[-1])
        ],dim=-1)

        return torch.tril(tmp.view(*self.batch_shape,n,n))
    
    def _eval_covar_matrix(self):
        cf = self.chol_factor
        return cf @ cf.transpose(-1,-2)
    
    @property
    def covar_matrix(self):
        return PsdSumLinearOperator(RootLinearOperator(self.chol_factor))

    def forward(self,i1,i2,**params):
        i1, i2 = i1.long(),i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = torch.broadcast_shapes(i1.shape[:-2],i2.shape[:-2],self.batch_shape)

        res = InterpolatedLinearOperator(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res

