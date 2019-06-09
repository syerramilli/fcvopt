import numpy as np

class LCB:
    """
    Lower confidence bound
    
    Parameters
    -------------
    model: GP or AGP object or any object which implements a 
    `predict` method with the following arguments
        - X
        - scaled
        - return_std
        
    kappa: float, optional (default: 2)
        Multiplier which constrols the exploration-exploitation tradeoff
    """
    def __init__(self,model,kappa=2):
        self.model = model
        self.kappa = kappa
        
    def update(self,model):
        self.model = model
        
    def __call__(self,x,scaled=True):
        x_copy = np.copy(x)
        if x.ndim == 1:
            x_copy = x_copy.reshape((1,-1))
        y_mean,y_std = self.model.predict(x_copy,scaled=scaled,return_std=True)
        val = y_mean - self.kappa*y_std
        return val