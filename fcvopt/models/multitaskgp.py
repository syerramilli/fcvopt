import torch
import gpytorch

from gpytorch.models import ExactGP
from gpytorch.constraints import GreaterThan,Positive
from gpytorch.priors import NormalPrior,LogNormalPrior
from ..priors import HalfHorseshoePrior,LogUniformPrior
from ..kernels import MultiTaskKernel

from typing import List

class MultitaskGPModel(ExactGP):
    '''
    Multi-task GP model used by Swersky et.al.
    '''
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        num_tasks:int,
        correlation_kernel_class,
        noise:float=1e-4,
        fix_noise:bool=False,
        estimation_method:str='MAP'
    ) -> None:

        # initializing likelihood
        noise_constraint=GreaterThan(1e-8)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)

        # standardizing the response variable
        y_mean,y_std = train_y.mean(),train_y.std()
        train_y_sc = (train_y-y_mean)/y_std

        # initializing ExactGP
        super().__init__(train_x,train_y_sc,likelihood)

        # registering mean and std of the raw response
        self.register_buffer('y_mean',y_mean)
        self.register_buffer('y_std',y_std)
        self.register_buffer('num_tasks',torch.tensor(num_tasks))

        # initializing and fixing noise
        if noise is not None:
            self.likelihood.initialize(noise=noise)

        if fix_noise:
            self.likelihood.raw_noise.requires_grad_(False)
        
        # Modules
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = correlation_kernel_class(
            ard_num_dims=self.train_inputs[0].size(1),
            lengthscale_constraint=Positive(),
        )
        self.task_covar_module = MultiTaskKernel(num_tasks)

        # priors
        if not fix_noise:
            noise_prior = LogUniformPrior(1e-8,2.) if estimation_method == 'MAP' \
                else HalfHorseshoePrior(0.1)
            self.likelihood.register_prior('noise_prior',noise_prior,'noise')

        self.mean_module.register_prior('mean_prior',NormalPrior(0.,1.),'constant')
        self.covar_module.register_prior('lengthscale_prior',LogUniformPrior(0.01,10.),'lengthscale')

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
    def reset_parameters(self):
        # reset the raw cholesky vector from N(0,1) 
        torch.nn.init.normal_(self.task_covar_module.raw_chol_vec,0.,1.)

        # sample the hyperparameters from their respective priors
        # Note: samples in place
        for _,prior,closure,setting_closure in self.named_priors():
            num_samples = (1,) if len(prior.shape()) > 0 else closure().shape
            setting_closure(prior.sample(num_samples))
    
    def predict(self,x,i=None,return_std=False,marginalize=False):
        '''
        Returns the prediction mean and variance at the given points
        # if i is None, then return the mean and std of the averaged estimate
        # at a single x
        '''
        self.eval()

        if i is None:
            # for compatibility with BayesOpt methods
            # return the mean at a single x
            i = torch.arange(self.num_tasks)
            x2 = x.repeat(self.num_tasks,1)
            output = self(x2,i)
            out_mean = output.mean.mean()*self.y_std + self.y_mean
            if return_std:
                out_covar = output.covariance_matrix*self.y_std**2
                out_std = out_covar.sum().sqrt()/self.num_tasks
                return out_mean,out_std

            return out_mean
        
        output = self(x,i)
        out_mean = output.mean*self.y_std + self.y_mean
        if return_std:
            out_std = output.variance.sqrt()*self.y_std
            return out_mean,out_std
        
        return out_mean