import torch
import gpytorch
from typing import Optional,Tuple,Union
from ..models.gpregression import GPR
from ..acquisition.acquisition import AcquisitionFunction

class PosteriorVariance(AcquisitionFunction):
    def __init__(self,model:GPR) -> None:
        super().__init__(model=model,maximize=True)
    
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        return self.model.return_var(X)

class ActiveLearningCohn(AcquisitionFunction):
    def __init__(self,model:GPR,X_ref) -> None:
        super().__init__(model=model,maximize=True)
        # precompute quantities not dependent on x_{n+1}
        Knref = model.covar_module(model.train_inputs[0],X_ref).evaluate()
        com_term = model.prediction_strategy.lik_train_train_covar.inv_matmul(Knref)
        self.register_buffer('com_term',com_term)
        self.register_buffer('X_ref',X_ref)
    
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        den = self.model.return_var(X)
        Knew_ref = self.model.covar_module(X,self.X_ref).evaluate()
        prod1 = self.model.covar_module(X,self.model.train_inputs[0]).matmul(self.com_term)

        term1 = prod1**2
        term2 = -2*prod1*Knew_ref
        term3 = Knew_ref**2

        impv = (term1+term2+term3)/den
        return impv.mean()