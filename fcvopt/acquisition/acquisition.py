import torch
import gpytorch
from typing import List

class AcquisitionFunction(torch.nn.Module):
    def __init__(self,model:gpytorch.models.ExactGP) -> None:
        super().__init__()
        self.add_module('model',model)
    
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        pass

class AcquisitionFunctionMCMC(AcquisitionFunction):
    def __init__(self,model:gpytorch.models.ExactGP) -> None:
        super().__init__(model)
        self.num_samples = len(model.train_inputs[0])

class MCAcquisitionFunctionWrapper(AcquisitionFunction):
    def __init__(self, model:gpytorch.models.ExactGP,base_acq:AcquisitionFunction,**kwargs) -> None:
        torch.nn.Module.__init__(self)
        #self.add_module('model',model)
        self.add_module('base_acq',base_acq(model=model,**kwargs))
        self.num_samples = len(self.base_acq.model.train_inputs[0])
    
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        return self.base_acq(X.unsqueeze(0).repeat(self.num_samples,1,1)).mean(axis=0)


