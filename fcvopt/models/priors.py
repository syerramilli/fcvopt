import math
import torch
from torch.nn import Module as TModule
from gpytorch.priors import Prior
from torch.distributions import constraints,HalfCauchy, Beta
from torch.distributions.utils import broadcast_all
from numbers import Number

class HalfCauchyPrior(Prior, HalfCauchy):
    """
    Half-Cauchy prior.
    """

    def __init__(self, scale, validate_args=None, transform=None):
        TModule.__init__(self)
        HalfCauchy.__init__(self, scale=scale, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        return HalfCauchyPrior(self.scale.expand(batch_shape))

class BetaPrior(Prior, Beta):
    """ Beta Prior parameterized by concentration parameters

    pdf(x) = 1/Beta(alpha,beta) * x^(alpha - 1) * (1-x)^(beta-1)

    were alpha > 0 and beta > 0 are the two concentration parameters, respectively.
    """
    support=constraints.interval(0.,1.)
    def __init__(self, concentration1, concentration0, validate_args=False, transform=None):
        TModule.__init__(self)
        Beta.__init__(self, concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return BetaPrior(self.concentration1.expand(batch_shape), self.concentration0.expand(batch_shape))

    def __call__(self, *args, **kwargs):
        return super(Beta, self).__call__(*args, **kwargs)

