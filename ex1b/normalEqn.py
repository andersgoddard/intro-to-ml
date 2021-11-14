import numpy as np
from numpy.linalg import inv

def normalEqn(X, y):
    mu = np.zeros((1, np.shape(X)[1]))

    theta = np.zeros((1, np.shape(X)[1]))
    
    a = np.transpose(X).dot(X)
    b = np.transpose(X).dot(y)
    
    theta = inv(a).dot(b)
    
    return theta