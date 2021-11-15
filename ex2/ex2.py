import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from plotData import *
from costFunction import *
from plotDecisionBoundary import *

#  Machine Learning Online Class - Exercise 2: Logistic Regression
# 
#  Instructions
#  ------------
#  
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
# 
#      sigmoid.m
#      costFunction.m
#      predict.m
#      costFunctionReg.m
# 
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
# 

#  Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = pd.read_csv('ex2data1.txt', header=None)
X = data.iloc[:, [0, 1]]
y = data.iloc[:, 2]

#  ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

plotData(X, y)

a = X

input('\nProgram paused. Press enter to continue.\n')

#  Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m = np.shape(X)[0]
n = np.shape(X)[1]

# Add intercept term to X
ones = np.ones((m, 1)).flatten()     # Create a column of ones
X = np.column_stack([ones, X])       # Add the column to X

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): ' + str(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

input('\nProgram paused. Press enter to continue.\n')

#  Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  using fmin_tnc from scipy
result = opt.fmin_tnc(func=costFunction, x0=initial_theta, args=(X, y))
optimal_theta = result[0]
cost, grad = costFunction(optimal_theta, X, y)

# Print cost and theta to screen
print('\nCost at theta found by fminunc: ' + str(cost))
print('Expected cost (approx): 0.203\n')
print('theta: ')

for i in optimal_theta:
    print(i)

print('\nExpected theta (approx): ')
print(' -25.161\n 0.206\n 0.201\n')

# Plot boundary
plotDecisionBoundary(optimal_theta, X, y, a)
