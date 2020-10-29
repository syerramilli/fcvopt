import torch
import gpytorch

from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel
from gpytorch.constraints import GreaterThan,Positive
from gpytorch.priors import NormalPrior
from fcvopt.priors import HalfHorseshoePrior
from typing import List

class GPR(ExactGP):
    '''
    The standard GP regression class deriving from `gpytorch.models.ExactGP`
    '''
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        correlation_kernel,
        noise:float=1e-4,
        fix_noise:bool=False
    ) -> None:
    
        # initializing likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=GreaterThan(1e-6,transform=torch.exp,inv_transform=torch.log),
        )

        # standardizing the response variable
        y_mean,y_std = train_y.mean(),train_y.std()
        train_y_sc = (train_y-y_mean)/y_std

        # initializing ExactGP
        super().__init__(train_x,train_y_sc,likelihood)

        # registering mean and std of the raw response
        self.register_buffer('y_mean',y_mean)
        self.register_buffer('y_std',y_std)

        # initializing and fixing noise
        if noise is not None:
            self.likelihood.initialize(noise=noise)

        if fix_noise:
            self.likelihood.raw_noise.requires_grad_(False)
        else:
            self.likelihood.register_prior('noise_prior',HalfHorseshoePrior(1.0),'noise')
        
        # Modules
        self.mean_module = gpytorch.means.ConstantMean(prior=NormalPrior(0.,1.))
        self.covar_module = ScaleKernel(
            correlation_kernel,
            outputscale_prior=gpytorch.priors.GammaPrior(2.,0.15)
        )
    
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)
    
    def predict(self,x,return_std=False,marginalize=True):
        '''
        Returns the predictive mean and variance at the given points
        '''
        self.eval()
        
        # determine if batched or not
        ndim = self.train_targets.ndim
        if ndim > 1:
            output = self(x)
        else:
            num_samples = self.train_targets.shape[0]
            output = self(x.unsqueeze(0).repeat(num_samples,1,1))
        
        out_mean = self.y_mean + self.y_std*output.mean

        # standard deviation may not always be needed
        if return_std:
            out_var = self.y_std*output.variance
            if (ndim > 1) and marginalize:
                # matching the second moment of the Gaussian mixture
                out_std = torch.sqrt(out_var.mean(axis=0)+out_mean.var(axis=0))
                out_mean = out_mean.mean(axis=0)
            else:
                out_std = out_var.sqrt() 
            return out_mean,out_std
        
        if ndim > 1 and marginalize:
            out_mean = out_mean.mean(axis=0)

        return out_mean