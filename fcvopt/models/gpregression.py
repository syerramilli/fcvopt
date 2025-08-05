import math
import torch
import gpytorch
from gpytorch import kernels
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.outcome import Standardize

from gpytorch.constraints import GreaterThan,Positive
from gpytorch.priors import NormalPrior,LogNormalPrior,GammaPrior

from .priors import HalfCauchyPrior
from .warp import InputWarp
from typing import Optional

def exp_with_shift(x:torch.Tensor):
    return 1e-6+x.exp()

class GPR(ExactGP, GPyTorchModel, FantasizeMixin):
    _num_outputs=1 # needed for botorch functions
    r'''Single output Gaussian Process Regression model.
     
    The model uses a Gaussian likelihood with a half-Cauchy prior on the noise level,
    and a constant mean function. The default covariance kernel is a Matern 5/2 kernel 
    with Automatic Relevance Determination (ARD) and a log-normal prior on the output scale. 
    The model can also be configured to accept custom covariance kernel objects, provided they 
    are extensions of `gpytorch.kernels.Kernel`.
    
    
    The model supports input warping via the `InputWarp` module, which can be enabled by setting 
    `warp_input=True`. This is useful for transforming the input space to improve model performance.
    
    Args:
        train_x: training data with dimensions (N x D).
        train_y: training targets with dimensions (N x 1).
        warp_input: If `True`, applies input warping to the input data. Default is `False`.
        covar_kernel: Optional custom covariance kernel object. If `None`, a default Matern 2/5 kernel is used.
    '''
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        warp_input:bool = False,
        covar_kernel:Optional[kernels.Kernel]= None
    ) -> None:
    
        # initializing likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            constraints=GreaterThan(1e-6,transform=torch.exp,inv_transform=torch.log)
        )

        outcome_transform = Standardize(1)
        train_y_sc,_ = outcome_transform(train_y.unsqueeze(-1))

        # initializing ExactGP
        super().__init__(train_x,train_y_sc.squeeze(-1),likelihood)

        # register outcome transform
        self.outcome_transform = outcome_transform

        # check if input warping is neeed
        self.register_buffer('warp_input',torch.tensor(warp_input))
        if self.warp_input:
            self.input_warping = InputWarp(input_dim = train_x.shape[-1])

        # Modules
        self.mean_module = gpytorch.means.ConstantMean()

        if covar_kernel is not None:
            self.covar_module = covar_kernel

        else:
            self.covar_module = kernels.ScaleKernel(
                base_kernel = kernels.MaternKernel(
                    ard_num_dims=self.train_inputs[0].size(1),
                    lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
                    lengthscale_prior=GammaPrior(3/2, 3.9/6),
                    nu=2.5
                ),
                outputscale_constraint=Positive(),
                outputscale_prior=LogNormalPrior(0., 1.)
            )  

        # register priors
        self.likelihood.register_prior('noise_prior',HalfCauchyPrior(0.1,transform=exp_with_shift),'raw_noise')
        self.mean_module.register_prior('mean_prior',NormalPrior(0.,1.),'constant')
    
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(self.input_warping(x) if self.warp_input else x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)
    
    def reset_parameters(self):
        # sample the hyperparameters from their respective priors
        # Note: samples in place
        for _,module,prior,closure,setting_closure in self.named_priors():
            if not closure(module).requires_grad:
                continue
            setting_closure(module,prior.expand(closure(module).shape).sample())

    def predict(self, x:torch.Tensor, return_std:bool=False) -> torch.Tensor | tuple[torch.Tensor]:
        r'''Returns the predicted mean and optionally the standard deviation of the model 
        at the given input points, conditioned on the training data.

        Args:
            x: Input points of shape (N x D)
            return_std: If `True`, also returns the standard deviation. Default is `False`.
        '''
        self.eval()

        out_dist = self.posterior(x).mvn
        out_mean = out_dist.loc

        if return_std:
            out_std = out_dist.stddev
            return out_mean,out_std

        return out_mean