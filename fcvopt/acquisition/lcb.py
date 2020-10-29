import torch
import gpytorch
from typing import Optional,Tuple,Union
from ..acquisition.acquisition import AcquisitionFunction, AcquisitionFunctionMCMC
from ..models.gpregression import GPR

class LowerConfidenceBound(AcquisitionFunction):
    def __init__(
        self,
        model:gpytorch.models.ExactGP,
        kappa:Union[float,torch.Tensor]
    ) -> None:
        super().__init__(model=model)
        if not torch.is_tensor(kappa):
            kappa = torch.tensor(kappa)
        self.register_buffer('kappa',kappa)
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        kappa = self.kappa.to(X)
        posterior = self.model(X)
        return posterior.mean - kappa*posterior.variance.sqrt()

class LowerConfidenceBoundMCMC(AcquisitionFunctionMCMC):
    def __init__(
        self,
        model:gpytorch.models.ExactGP,
        kappa:Union[float,torch.Tensor]
    ) -> None:
        super().__init__(model=model)
        if not torch.is_tensor(kappa):
            kappa = torch.tensor(kappa)
        self.register_buffer('kappa',kappa)
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        kappa = self.kappa.to(X)
        
        # matched first and second moments
        mu,sigma = self.model.predict(X,return_std=True,marginalize=True)

        return mu - kappa*sigma