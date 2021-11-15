import numpy as np
from numpy import zeros_like, asarray
from scipy.optimize import fmin_ncg
from sigmoid import *

def costFunction(theta, X, y):
    m = len(y)
    J = 0
    grad = np.zeros((np.shape(theta)[0], 1))    

    
    h = sigmoid(X.dot(theta)).flatten()
    J = (1/m) * sum((-y * np.log(h)) - ((1 - y) * np.log(1-h)))
    grad = (1/m) * (np.transpose(X).dot(h - y))
    
    return J, grad.flatten()
    