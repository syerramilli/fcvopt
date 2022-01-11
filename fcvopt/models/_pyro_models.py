import math
import pyro
import torch
import pyro.distributions as dist

from gpytorch.lazy import lazify
from gpytorch.distributions import MultivariateNormal
from ..priors import HalfHorseshoePrior,LogUniformPrior,BetaPrior

class MVN(dist.MultivariateNormal):
    def __init__(self,base_dist,added_loss_terms=None):
        self.base_dist = base_dist
    
    def rsample(self,**kwargs):
        return self.base_dist.rsample(**kwargs)
    
    def log_prob(self,value):
        try:
            out= self.base_dist.log_prob(value)
            return out
        except Exception as e:
            raise RuntimeError("singular U")
    
    @property
    def event_shape(self):
        return self.base_dist.event_shape
    
    @property
    def _batch_shape(self):
        return self.base_dist._batch_shape


def _sq_dist(x1,x2=None):
    if x2 is None:
        x2 = x1
    
    x1_sq = (x1**2).sum(1,keepdim=True)
    x2_sq = (x2**2).sum(1,keepdim=True)
    x1x2 = x1.matmul(x2.t()) 
    r2 = x1_sq + x2_sq.t() - 2*x1x2
    return r2.clamp(min=0)

def rbf_kernel(x1,x2=None):
    return torch.exp(-0.5*_sq_dist(x1,x2))

def matern52_kernel(x1,x2=None):
    r2 = _sq_dist(x1,x2)
    # add buffer to avoid the NaN gradient issue of torch.sqrt at 0.
    term1 = torch.sqrt(5*r2+1e-12)
    return (1+ term1 + (5/3) *r2)*torch.exp(-term1)

def hamming_kernel(rho,x1,x2=None):
    if x2 is None:
        x2 = x1
    
    dist = torch.cdist(x1,x2,p=0.)
    return 1.-dist.mul(1.-rho)

def pyro_gp(x,y,jitter):
    mean = pyro.sample('mean_module.mean_prior',dist.Normal(0,1).expand([1]))
    outputscale = pyro.sample("covar_module.outputscale_prior", dist.LogNormal(0., 1.))
    noise = pyro.sample(
        "likelihood.noise_prior",HalfHorseshoePrior(0.1).expand([1])
    )
    
    lengthscale = pyro.sample(
        'covar_module.base_kernel.lengthscale_prior',
        LogUniformPrior(0.01,10.).expand([1,x.shape[1]]).to_event(2)
    )

    x2 = x/lengthscale
    
    # compute kernel
    k = outputscale*matern52_kernel(x2)
    # add noise and jitter
    k += (noise+jitter)*torch.eye(x.shape[0])
    
    # sample Y according to the standard gaussian process formula
    pyro.sample(
        "y",
        MVN(
            MultivariateNormal(
                mean*torch.ones(x.shape[0]).to(x), lazify(k.to(x))
            )
        ),
        obs=y,
    )

def pyro_hgp(x_aug,y,jitter):
    x = x_aug[0]
    fold_idxs = x_aug[1]
    mean = pyro.sample('mean_module.mean_prior',dist.Normal(0,1).expand([1]))

    noise = pyro.sample(
        "likelihood.noise_prior",HalfHorseshoePrior(0.1).expand([1])
    )
    
    outputscale_f = pyro.sample("covar_module.outputscale_prior", dist.LogNormal(0., 1.))
    lengthscale_f = pyro.sample(
        'covar_module.base_kernel.lengthscale_prior',
        LogUniformPrior(0.01,10.).expand([1,x.shape[1]]).to_event(2)
    )

    outputscale_delta = pyro.sample("covar_module_delta.outputscale_prior", dist.LogNormal(-2., 2.))
    lengthscale_delta = pyro.sample(
        'covar_module_delta.base_kernel.lengthscale_prior',
        LogUniformPrior(0.01,10.).expand([1,x.shape[1]]).to_event(2)
    )
    rho = pyro.sample(
        'corr_delta_fold.rho_prior',
        BetaPrior(1.,5.).expand([1,1])
    )

    x2_f = x/lengthscale_f
    x2_delta = x/lengthscale_delta
    
    # compute kernel
    k = outputscale_f*matern52_kernel(x2_f)
    k += (outputscale_delta*matern52_kernel(x2_delta)).mul(hamming_kernel(rho, fold_idxs))
    # add noise and jitter
    k += (noise+jitter)*torch.eye(x.shape[0])
    
    # sample Y according to the standard gaussian process formula
    pyro.sample(
        "y",
        MVN(
            MultivariateNormal(
                mean*torch.ones(x.shape[0]).to(x), lazify(k.to(x))
            )
        ),
        obs=y,
    )