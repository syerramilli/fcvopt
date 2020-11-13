import torch
import torch.distributions as dist
#from torch.nn import Module as TModule
from gpytorch.priors import Prior

class InverseGamma(dist.TransformedDistribution):
    r"""
    Creates an inverse-gamma distribution parameterized by
    `concentration` and `rate`.

        X ~ Gamma(concentration, rate)
        Y = 1/X ~ InverseGamma(concentration, rate)

    :param torch.Tensor concentration: the concentration parameter (i.e. alpha).
    :param torch.Tensor rate: the rate parameter (i.e. beta).
    """
    def __init__(self, concentration, rate):
        base_dist = dist.Gamma(concentration,rate)
        super(InverseGamma, self).__init__(
            base_dist,
            dist.PowerTransform(-torch.ones_like(base_dist.rate)))

class InverseGammaPrior(Prior, InverseGamma):
    """
    Log Uniform prior.
    """

    def __init__(self, concentration,rate):
        #TModule.__init__(self)
        InverseGamma.__init__(self, concentration,rate)
        self._transform = None

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return InverseGammaPrior(
            self.base_dist.concentration.expand(batch_shape), 
            self.base_dist.rate.expand(batch_shape))