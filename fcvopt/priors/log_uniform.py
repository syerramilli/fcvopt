import torch
import torch.distributions as dist
#from torch.nn import Module as TModule
from gpytorch.priors import Prior

class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        if not torch.is_tensor(lb):
            lb = torch.tensor(lb)
        
        if not torch.is_tensor(ub):
            ub = torch.tensor(ub)

        super(LogUniform, self).__init__(dist.Uniform(lb.log(), ub.log()),
                                         dist.ExpTransform())

class LogUniformPrior(Prior, LogUniform):
    """
    Log Uniform prior.
    """

    def __init__(self, a, b):
        #TModule.__init__(self)
        LogUniform.__init__(self, a, b)
        self._transform = None

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return LogUniformPrior(
            self.base_dist.low.exp().expand(batch_shape), 
            self.base_dist.high.exp().expand(batch_shape))