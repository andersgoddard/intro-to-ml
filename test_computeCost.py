import pytest
import numpy as np
import pandas as pd
from computeCost import *

def test_computeCost():

    data = pd.read_csv('ex1data1.csv', header=None)

    X = data.iloc[:, 0]
    y = data.iloc[:, 1]
    m = len(y)
    
    ones = np.ones((m, 1)).flatten()
    X = np.column_stack([ones, X])
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    J = computeCost(X, y, theta)
    expectedJapprox = 32.07
    
    assert abs(J - expectedJapprox) < 0.01 
    
def test_computeCost_2():

    data = pd.read_csv('ex1data1.csv', header=None)

    X = data.iloc[:, 0]
    y = data.iloc[:, 1]
    m = len(y)
    
    ones = np.ones((m, 1)).flatten()
    X = np.column_stack([ones, X])
    theta = np.array([[-1], [2]])
    iterations = 1500
    alpha = 0.01
    J = computeCost(X, y, theta)
    expectedJapprox = 54.24
    
    assert abs(J - expectedJapprox) < 0.01 
    
    