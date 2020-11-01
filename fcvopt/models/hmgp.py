import torch
import gpytorch

from gpytorch.kernels import ScaleKernel
from gpytorch.constraints import GreaterThan,Positive
from gpytorch.priors import LogNormalPrior
from .gpregression import GPR
from ..priors import LogUniformPrior
from ..kernels import HammingKernel

# for extracting predictions
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch.utils.broadcasting import _mul_broadcast_shape

from typing import List

class HGP(GPR):
    '''
    HGP model
    '''
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        correlation_kernel_class,
        noise:float=1e-4,
        fix_noise:bool=False
    ) -> None:
        super().__init__(
            train_x=train_x,train_y=train_y,
            correlation_kernel_class=correlation_kernel_class,
            noise=noise,fix_noise=fix_noise
        )

        # similar to f
        self.covar_module_delta = ScaleKernel(
            base_kernel = correlation_kernel_class(
                ard_num_dims=train_x[0].size(1),
                lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
                lengthscale_prior=LogUniformPrior(0.01,10.)
            ),
            outputscale_prior=LogNormalPrior(-1.,1.),
            outputscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log)
        )

        self.corr_delta_fold = HammingKernel()
    
    def forward(self,x,fold_idx):
        mean_x = self.mean_module(x)
        covar_f = self.covar_module(x)
        covar_delta = self.covar_module_delta(x).mul(self.corr_delta_fold(fold_idx))
        covar = covar_f + covar_delta
        return gpytorch.distributions.MultivariateNormal(mean_x,covar)
    
    def forward_f(self,x):
        # Only with GP f
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)
    
    def _predict(self,*args):
        # the prediction routine is different from GPR in that we are interested only
        # in the main GP `f` and not delta. 
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
    
    def _fold_selection_metric(self,x,fold_idxs):
        # This method is for internal use only!.
        #TODO: raise error if self.prediction_strategy is None
        #TODO: raise error if not in eval model

        # determine if batched or not
        ndim = self.train_targets.ndim
        if ndim == 1:
            x_new = x
        else:
            num_samples = self.train_targets.shape[0]
            x_new = x.unsqueeze(0).repeat(num_samples,1,1)
        
        ## precompute quantities common across all folds
        covar_f_new = self.covar_module(self.train_inputs[0],x_new).evaluate()
        covar_delta_new_x = self.covar_module_delta(self.train_inputs[0],x_new)
        term1 = self.prediction_strategy.lik_train_train_covar.inv_matmul(covar_f_new)

        # compute the fold selection criterion for all the folds
        out = []
        for fold_idx in fold_idxs:
            fold_new = torch.tensor([[fold_idx]]).to(self.train_inputs[1])
            if ndim > 1:
                fold_new = fold_new.unsqueeze(0).repeat(num_samples,1,1)
            
            covar_delta_new = covar_delta_new_x.mul(
                self.corr_delta_fold(self.train_inputs[1],fold_new)
            ).evaluate()
            
            term4 = (covar_f_new+covar_delta_new).transpose(-1,-2).matmul(term1).flatten()
            out.append(
                term4.mul(term4-self.covar_module.outputscale).mean().item()
            )
        
        return out