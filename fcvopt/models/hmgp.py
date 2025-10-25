import torch
import gpytorch

from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Positive
from gpytorch.priors import LogNormalPrior, GammaPrior, HalfCauchyPrior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings
from .gpregression import GPR
from ..kernels import HammingKernel

# for extracting predictions
from gpytorch.models.exact_prediction_strategies import prediction_strategy

from typing import List

class HGP(GPR):
    r'''Hierarchical GP model for modeling the CV loss function

    This model is a sum of a main GP `f`, that models the CV loss function, and
    a delta GP that models the deviation of the individual fold holdout losses 
    from the CV loss.

    The model is defined as

    .. math::

        \begin{equation*}
            y_j(x) = f(x) + \delta_j(x) + \epsilon_j(x)
        \end{equation*}

    where :math:`\delta_j(x)` is the deviation of the individual fold holdout losses
    from the CV loss, and :math:`\epsilon_j(x)` is the observation noise.

    Args:
        train_x: training data with dimensions (N x (D + 1)) where the last column 
            contains the fold indices
        train_y: training targets with dimensions (N x 1)
        warp_input: whether to apply input warping to the inputs. Default: False
    '''
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        warp_input:bool=False
    ) -> None:
        super().__init__(
            train_x=train_x,train_y=train_y,
            warp_input=warp_input
        )

        # similar to f
        self.covar_module_delta = ScaleKernel(
            base_kernel = MaternKernel(
                nu=2.5,
                ard_num_dims=train_x[0].size(1),
                lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
                lengthscale_prior=GammaPrior(3/2, 3.9/6)
            ),
            outputscale_prior= LogNormalPrior(-1., 2.),
            outputscale_constraint=Positive(
                initial_value=torch.tensor(0.1)
            )
        )

        self.corr_delta_fold = HammingKernel()

    def forward(self, x:torch.Tensor, fold_idx:torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        r'''Forward pass  of the model for :math:`y_{fold\_idx}(x)` at hyperparameters x 
        and fold index `fold_idx`

        Args:
            x: A tensor of input locations with dimensions (N x D)
            fold_idx: A tensor of fold indices with dimensions (N x 1)
        '''
        if self.warp_input:
            x = self.input_warping(x)

        mean_x = self.mean_module(x)
        covar_f = self.covar_module(x)
        covar_delta = self.covar_module_delta(x).mul(self.corr_delta_fold(fold_idx))
        covar = covar_f + covar_delta
        return gpytorch.distributions.MultivariateNormal(mean_x,covar)
    
    def forward_f(self, x:torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        r'''
        Forward pass of the main GP `f` for :math:`f(x)` at x

        Args:
            x: A tensor of input locations with dimensions (N x D)
        '''
        # Only with GP f
        if self.warp_input:
            x = self.input_warping(x)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)

    def posterior(
        self, X:torch.Tensor,
        observation_noise:bool=False,
        **kwargs
    ) -> GPyTorchPosterior:
        '''Returns the posterior distribution of the CV loss :math:`f(\cdot)` at the input locations `X`

        .. note::

            This method returns the posterior distribution of the main GP `f`, not  
            the posterior distribution of the individual fold holdout losses.

        Args:
            X: input locations with dimensions (N x D). Note that this should not include
                the fold indices.
            observation_noise: whether to include the observation noise in the posterior.
                We recommend setting this to False. Default: False
            kwargs: not used
        '''
        self.eval()  # make sure model is in eval mode
        
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)
        with gpt_posterior_settings():
            mvn = self._call_f(X)

            if observation_noise:
                mvn = self.likelihood(mvn, X)

        posterior = GPyTorchPosterior(distribution=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        
        return posterior
    
    def _call_f(self,*args):
        # the prediction routine is different from GPR in that we are interested only
        # in the main GP `f` and not delta. 
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args][0]

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
        train_input = train_inputs[0]
        # Make sure the batch shapes agree for training/test data
        if batch_shape != train_input.shape[:-2]:
            batch_shape = torch.broadcast_shapes(batch_shape, train_input.shape[:-2])
            train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
        if batch_shape != inputs.shape[:-2]:
            batch_shape = torch.broadcast_shapes(batch_shape, inputs.shape[:-2])
            train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
            inputs = inputs.expand(*batch_shape, *inputs.shape[-2:])
        full_inputs.append(torch.cat([train_input, inputs], dim=-2))

        # Get the joint distribution for training/test data
        full_output = self.forward_f(*full_inputs)
        full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

        # Determine the shape of the joint distribution
        batch_shape = full_output.batch_shape
        joint_shape = full_output.event_shape
        tasks_shape = joint_shape[1:]  # For multitask learning
        test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

        # Make the prediction
        predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

        # Reshape predictive mean to match the appropriate event shape
        predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
        return full_output.__class__(predictive_mean, predictive_covar)
    
    def _fold_selection_metric(self,x,fold_idxs):
        # This method is for internal use only!.
        #TODO: raise error if self.prediction_strategy is None
        #TODO: raise error if not in eval mode

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

    def condition_on_observations(self, X:torch.Tensor, Y:torch.Tensor, **kwargs) -> GPR:
        r'''
        Returns a new model for the true CV loss function :math:`f(\cdot)` 
        conditioned on new observations :math:`Y` at the input locations :math:`X`. 
        
        This is used for fantasy modeling for the knowledge gradient and batch acquistion
        functions, where we want to obtain a new model with these observations added to 
        the existing training data without modifying the original model. 

        Args:
            X: input locations with dimensions (N x D), where N is the number of observations
                and D is the number of input dimensions.
            Y: targets at the input locations with dimensions (N x 1)
            **kwargs: not used here
        '''

        Yvar = kwargs.get('noise', None)

        if hasattr(self, "outcome_transform"):
            # pass the transformed data to get_fantasy_model below
            Y,Yvar = self.outcome_transform(Y,Yvar)

        if Y.size(-1)==1:
            Y = Y.squeeze(-1)
        
        # Create a new GP model based on the CV loss f
        # Note: we need to pass the untransformed targets to the new model
        # because the same outcome transform will be applied when creating
        # the GPR model
        new_model = GPR(
            train_x = self.train_inputs[0],
            train_y = self.outcome_transform.untransform(self.train_targets)[0].flatten(),
            covar_kernel= self.covar_module
        ).double()

        _ = new_model.initialize(**{
            'likelihood.noise_covar.raw_noise':self.likelihood.noise_covar.raw_noise.detach().clone(),
            'mean_module.raw_constant':self.mean_module.raw_constant.detach().clone()
        })

        _ = new_model.eval()
        # copy the prediction cache of the original model to this new model
        new_model.prediction_strategy = self.prediction_strategy

        # get_fantasy_model will properly copy any existing outcome transforms
        # (since it deepcopies the original model)
        return new_model.get_fantasy_model(inputs=X,targets=Y)
        
