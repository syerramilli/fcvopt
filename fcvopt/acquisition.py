import numpy as np

class LCB:
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