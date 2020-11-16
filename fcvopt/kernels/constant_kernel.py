import torch
from gpytorch.kernels import Kernel

class ConstantKernel(Kernel):
    has_lengthscale=False

    def forward(self,x1,x2,diag=False,**params):
        if diag:
            torch.ones(*self.batch_shape,x1.shape[-2]).to(x1)
        
        return torch.ones(*self.batch_shape,x1.shape[-2],x2.shape[-2]).to(x1)