# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from fcvopt.optimizers.fcvopt import FCVOpt
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.preprocessing import normalize

dat = pd.read_csv('data/spambase.data',header=None)
X = dat.values[:,:-1]
y = dat.values[:,-1]
X = normalize(X)

param_bounds = {'C':[2**-5,2**15],'gamma':[2**-15,2**5]}
opt = FCVOpt(SVC(probability=True),param_bounds,log_loss,kernel="gaussian",
             logscale=np.array([0,1]),max_iter=5,seed=2,verbose=2)
opt.fit(X,y)