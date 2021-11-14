import pytest
import numpy as np
import pandas as pd
from computeCost import *
from gradientDescent import *

def test_gradientDescent():

    data = pd.read_csv('ex1data1.csv', header=None)

    X = data.iloc[:, 0]
    y = data.iloc[:, 1]
    m = len(y)
    
    ones = np.ones((m, 1)).flatten()
    X = np.column_stack([ones, X])
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    
    theta = gradientDescent(X, y, theta, alpha, iterations)
    expectedTheta0 = -3.6303
    expectedTheta1 = 1.1664
    
    assert abs(theta.flat[0] - expectedTheta0) < 0.01 
    assert abs(theta.flat[1] - expectedTheta1) < 0.01 
