import numpy as np

def rmse(y_true, y_pred):
    '''
    Computes root-mean squared error
    '''
    return np.sqrt(np.mean((y_true-y_pred)**2))

def mcr(y_true,y_pred):
    '''
    Computes misclassification rate
    '''
    return 1-np.mean(y_true==y_pred)