import numpy as np
from math import e

def sigmoid(z):
    g = np.zeros((np.shape(z)[0], 1))
    
    g = 1 / (1 + pow(e, -z))
    
    return g