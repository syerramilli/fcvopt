import numpy as np

class BaseModel(object):
    '''
    Base class for all models
    '''
    def __init__(self):
        self.X = None
        self.y = None
        self.n_dim = None
        self.is_trained = False
    
    def fit(self,X,y):
        pass
    
    def update(self,X,y):
        self.X = np.append(self.X,X,axis=0)
        self.y = np.append(self.y,y,axis=0)
        return self.fit(X,y)
    
    def predict(self,X):
        pass
    
    def get_incumbent(self):
        pass
