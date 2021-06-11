import torch
import gpytorch
from typing import Optional,Tuple,Union
from ..models.gpregression import GPR
from ..acquisition.acquisition import AcquisitionFunction
from torch.distributions import Normal

class ExpectedImprovement(AcquisitionFunction):
    def __init__(
        self,
        model:GPR,
        f_best:Union[float,torch.Tensor]
    ) -> None:
        super().__init__(model=model,maximize=True)
        if not torch.is_tensor(f_best):
            f_best = torch.tensor(f_best)

        self.register_buffer('f_best',f_best)
    
    def forward(self,X:torch.Tensor,**kwargs) -> torch.Tensor:
        f_best = self.f_best.to(X)
        mu,sigma = self.model.predict(X,return_std=True,marginalize=False,**kwargs)
        u = (f_best.expand_as(mu)-mu)/sigma
        normal = Normal(torch.zeros_like(u),torch.ones_like(u))
        
        return sigma*(torch.exp(normal.log_prob(u)) + u*normal.cdf(u))
