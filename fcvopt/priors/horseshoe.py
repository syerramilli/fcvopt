import math
import torch
from numbers import Number
from gpytorch.priors import Prior
from torch.distributions import HalfCauchy, HalfNormal, constraints


class HalfHorseshoePrior(Prior):
    """Horseshoe prior.

    There is no analytical form for the horeshoe prior's pdf, but it
    satisfies a tight bound of the form `lb(x) <= pdf(x) <= ub(x)`, where

        lb(x) = K/2 * log(1 + 4 * (scale / x) ** 2)
        ub(x) = K * log(1 + 2 * (scale / x) ** 2)

    with `K = 1 / sqrt(2 pi^3)`. Here, we simply use

        pdf(x) ~ (lb(x) + ub(x)) / 2

    Reference: C. M. Carvalho, N. G. Polson, and J. G. Scott.
        The horseshoe estimator for sparse signals. Biometrika, 2010.
    """

    arg_constraints = {"scale": constraints.positive}
    support = constraints.positive
    _validate_args = True

    def __init__(self, scale, validate_args=False):
        #TModule.__init__(self)
        if isinstance(scale, Number):
            scale = torch.tensor(float(scale))
        self.K = 1 / math.sqrt(2 * math.pi ** 3)
        self.scale = scale
        super().__init__(scale.shape, validate_args=validate_args)
        # now need to delete to be able to register buffer
        del self.scale
        self.register_buffer("scale", scale)
        self._transform = None

    def log_prob(self, X):

        A = (self.scale / self.transform(X)) ** 2
        lb = self.K / 2 * torch.log(1 + 4 * A)
        ub = self.K * torch.log(1 + 2 * A)
        return torch.log((lb + ub) / 2)

    def rsample(self, sample_shape=torch.Size([])):
        local_shrinkage = HalfCauchy(1).rsample(self.scale.shape)
        param_sample = HalfNormal(local_shrinkage * self.scale).rsample(sample_shape)
        param_sample[param_sample<1.01e-6] = 1.01e-6
        return param_sample

    def expand(self,expand_shape, _instance=None):
        batch_shape = torch.Size(expand_shape)
        return HalfHorseshoePrior(self.scale.expand(batch_shape))