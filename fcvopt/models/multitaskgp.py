import torch
import gpytorch

from gpytorch.constraints import GreaterThan,Positive,Interval
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings
from .gpregression import GPR
from ..kernels import MultiTaskKernel, HammingKernel

from typing import List

class MultitaskGPModel(GPR):
    '''
    Multi-task GP model used by Swersky et.al.
    '''
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        num_tasks:int,
        warp_input:bool=False
    ) -> None:

        super().__init__(
            train_x=train_x,train_y=train_y,
            warp_input=warp_input
        )
        self.task_covar_module = MultiTaskKernel(num_tasks)
        self.register_buffer('num_tasks',torch.tensor(num_tasks))

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
        for _,module,prior,closure,setting_closure in self.named_priors():
            if not closure(module).requires_grad:
                continue
            setting_closure(module,prior.expand(closure(module).shape).sample())

    def posterior(
        self,X:torch.Tensor,i:torch.Tensor,
        observation_noise:bool=False,
        posterior_transform=None,**kwargs
    ):
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)

        with gpt_posterior_settings():
            mvn = self(X,i)

            if observation_noise:
                mvn = self.likelihood(mvn, X)
        
        posterior = GPyTorchPosterior(distribution=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        
        return posterior

    
    def predict(self,x,i=None,return_std=False):
        '''
        Returns the prediction mean and variance at the given points
        # if i is None, then return the mean and std of the averaged estimate
        # at a single x
        '''
        self.eval()

        if i is None:
            # for compatibility with BayesOpt methods
            # return the mean at a single x
            i = torch.tile(torch.arange(self.num_tasks),dims=(x.shape[0],)).unsqueeze(-1).long()
            x2 = x.repeat_interleave(self.num_tasks,dim=0)
            out_dist = self.posterior(x2,i).mvn
            out_mean = out_dist.loc.view(-1,self.num_tasks).mean(dim=-1) #output.mean.mean()*self.y_std + self.y_mean
            if return_std:
                # TODO: fix this for multiple xs
                out_covar = out_dist.covariance_matrix
                m = self.num_tasks
                n = x.shape[0]

                out_std = torch.tensor([
                    out_covar[idx:(idx+n),idx:(idx+n)].sum() for idx in range(0,m*n,n)
                ]).sqrt()/self.num_tasks
                return out_mean,out_std.clamp(1e-6)

            return out_mean
        
        out_dist = self.posterior(x,i).mvn
        out_mean = out_dist.loc

        if return_std:
            out_std = out_dist.stddev.clamp(1e-6)
            return out_mean,out_std
            
        return out_mean
    
class MultiTaskGPConstantCorrModel(MultitaskGPModel):
    def __init__(self, train_x, train_y, num_tasks, warp_input = False):
        super().__init__(train_x, train_y, num_tasks, warp_input)
        self.task_covar_module = HammingKernel()

    def reset_parameters(self):
        # sample the hyperparameters from their respective priors
        # Note: samples in place
        for _,module,prior,closure,setting_closure in self.named_priors():
            if not closure(module).requires_grad:
                continue
            setting_closure(module,prior.expand(closure(module).shape).sample())