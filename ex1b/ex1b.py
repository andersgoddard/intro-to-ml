import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from featureNormalize import *
from gradientDescentMulti import *
from normalEqn import *

#  Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
# 
#      warmUpExercise.m
#      plotData.m
#      gradientDescent.m
#      computeCost.m
#      gradientDescentMulti.m
#      computeCostMulti.m
#      featureNormalize.m
#      normalEqn.m
# 
#   For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
# 
#  x refers to the population size in 10,000s
#  y refers to the profit in $10,000s
# 


# Part 1: Feature Normalization
print('Loading data ...\n')
data = pd.read_csv('ex1data2.csv', header=None)

X = data.iloc[:, [0, 1]]
y = data.iloc[:, 2]
m = len(y)

# Print out some data points
print('First 10 examples from the dataset: \n')
print('X\n' + str(X[:10]) + '\n\ny\n' + str(y[:10]))

input('Program paused. Press enter to continue.\n')

print('Normalizing Features ...\n')

X, mu, sigma = featureNormalize(X)

ones = np.ones((m, 1)).flatten()     # Create a column of ones
X = np.column_stack([ones, X])       # Add the column to X

# Part 2: Gradient Descent

#  ====================== YOUR CODE HERE ======================
#  Instructions: We have provided you with the following starter
#                code that runs gradient descent with a particular
#                learning rate (alpha). 
# 
#                Your task is to first make sure that your functions - 
#                computeCost and gradientDescent already work with 
#                this starter code and support multiple variables.
# 
#                After that, try running gradient descent with 
#                different values of alpha and see which one gives
#                you the best result.
# 
#                Finally, you should complete the code at the end
#                to predict the price of a 1650 sq-ft, 3 br house.
# 
#  Hint: By using the 'hold on' command, you can plot multiple
#        graphs on the same figure.
# 
#  Hint: At prediction, make sure you do the same feature normalization.
# 

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1)) 
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
xrange = np.arange(1, num_iters+1, 1)
plt.plot(xrange, J_history.flatten())
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()


# Display gradient descent's result
print('Theta computed from gradient descent: \n');
print(str(theta));
print('\n');

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

x1 = 1
x2 = (1650 - mu[0]) / sigma[0]
x3 = (3 - mu[1]) / sigma[1]

xArray = np.array([x1, x2, x3])

price = xArray.dot(theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $' + str(price))

input('Program paused. Press enter to continue.\n')

# Part 3: Normal Equations ================

print('Solving with normal equations...\n')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form 
              # solution for linear regression using the normal
              # equations. You should complete the code in 
              # normalEqn.m

              # After doing so, you should complete this code 
              # to predict the price of a 1650 sq-ft, 3 br house.

# Load data
print('Loading data ...\n')
data = pd.read_csv('ex1data2.csv', header=None)

X = data.iloc[:, [0, 1]]
y = data.iloc[:, 2]
m = len(y)

# Add intercept term to X
ones = np.ones((m, 1)).flatten()     # Create a column of ones
X = np.column_stack([ones, X])       # Add the column to X

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(str(theta))
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
x1 = 1
x2 = (1650 - mu[0]) / sigma[0]
x3 = (3 - mu[1]) / sigma[1]

xArray = np.array([x1, x2, x3])

price = xArray.dot(theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equation): $' + str(price))

