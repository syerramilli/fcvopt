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
    
    def predict(self,X):
        pass
    
    def get_incumbent(self):
        pass