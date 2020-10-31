import torch
import gpytorch

from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel
from gpytorch.constraints import GreaterThan,Positive
from gpytorch.priors import NormalPrior,LogNormalPrior,GammaPrior
from ..models.gpregression import GPR
from ..priors import HalfHorseshoePrior,LogUniformPrior
from ..kernels import HammingKernel


# for extracting predictions
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch.utils.broadcasting import _mul_broadcast_shape

from typing import List


class HMGP(ExactGP):
    '''
    HMGP model
    '''
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        correlation_kernel_class,
        noise:float=1e-4,
        fix_noise:bool=False
    ) -> None:
    
        # initializing likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
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
            self.likelihood.register_prior('noise_prior',LogUniformPrior(1e-6,2.),'noise')
        
        # Modules
        self.mean_module = gpytorch.means.ConstantMean(prior=NormalPrior(0.,1.))

        # covariance modules - there are two modules one for f and one for delta
        self.covar_module = ScaleKernel(
            base_kernel = correlation_kernel_class(
                ard_num_dims=train_x[0].size(1),
                lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
                lengthscale_prior=LogUniformPrior(0.01,10.)
            ),
            outputscale_prior=LogNormalPrior(0.,1.),
            outputscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log)
        )

        self.corr_module_delta_x = correlation_kernel_class(
            ard_num_dims=train_x[0].size(1),
            lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
            lengthscale_prior=LogUniformPrior(0.01,10.)
        )
        
        self.covar_module_delta_fold = ScaleKernel(
            HammingKernel(),
            outputscale_prior=LogNormalPrior(-1.,1.),
            outputscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log)
        )
    
    def forward(self,x,fold_idx):
        mean_x = self.mean_module(x)
        covar_f = self.covar_module(x)
        covar_delta = self.covar_module_delta_fold(fold_idx).mul(self.corr_module_delta_x(x))
        covar = covar_f + covar_delta
        return gpytorch.distributions.MultivariateNormal(mean_x,covar)
    
    def forward_f(self,x):
        # Only with GP f
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
        if ndim == 1:
            output = self._predict(x) # notice the difference here
        else:
            num_samples = self.train_targets.shape[0]
            output = self._predict(x.unsqueeze(0).repeat(num_samples,1,1))
        
        out_mean = self.y_mean + self.y_std*output.mean

        # standard deviation may not always be needed
        if return_std:
            out_var = output.variance*self.y_std**2
            if (ndim > 1) and marginalize:
                # matching the second moment of the Gaussian mixture
                out_std = torch.sqrt(out_var.mean(axis=0)+out_mean.var(axis=0))
                out_mean = out_mean.mean(axis=0)
            else:
                out_std = out_var.sqrt() 
            return out_mean,out_std
        
        if (ndim > 1) and marginalize:
            out_mean = out_mean.mean(axis=0)

        return out_mean
    
    def _predict(self,*args):
        # returns the predictive mean

        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        # Get the terms that only depend on training data
        if self.prediction_strategy is None:
            train_output = self.forward(*train_inputs)

            # Create the prediction strategy for
            self.prediction_strategy = prediction_strategy(
                train_inputs=train_inputs,
                train_prior_dist=train_output,
                train_labels=self.train_targets,
                likelihood=self.likelihood,
            )
        
        # Concatenate the input to the training input
        # Note: includes only the xs
        full_inputs = []
        batch_shape = train_inputs[0].shape[:-2]
        for train_input, input in zip(train_inputs[0], inputs):
            # Make sure the batch shapes agree for training/test data
            if batch_shape != train_input.shape[:-2]:
                batch_shape = _mul_broadcast_shape(batch_shape, train_input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
            if batch_shape != input.shape[:-2]:
                batch_shape = _mul_broadcast_shape(batch_shape, input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                input = input.expand(*batch_shape, *input.shape[-2:])
            full_inputs.append(torch.cat([train_input, input], dim=-2))

        # Get the joint distribution for training/test data
        full_output = self.forward_f(*full_inputs)
        full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

        # Determine the shape of the joint distribution
        batch_shape = full_output.batch_shape
        joint_shape = full_output.event_shape
        tasks_shape = joint_shape[1:]  # For multitask learning
        test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

        # Make the prediction
        with gpytorch.settings._use_eval_tolerance():
            predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

        # Reshape predictive mean to match the appropriate event shape
        predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
        return full_output.__class__(predictive_mean, predictive_covar)