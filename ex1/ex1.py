import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from computeCost import *
from gradientDescent import *

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


# Part 1: Basic Function
print('Running warmUpExercise ...\n')
print('5 x 5 Identity Matrix: \n')

fiveByFive = np.identity(5)
              
print(fiveByFive)

input('Program paused. Press enter to continue.\n')


# Part 2: Plotting
data = pd.read_csv('ex1data1.csv', header=None)

initialX = data.iloc[:, 0]
y = data.iloc[:, 1]
m = len(y)

plt.scatter(initialX, y, marker='x')
plt.show()

input('Program paused. Press enter to continue.\n')


# Part 3: Cost and Gradient Descent
ones = np.ones((m, 1)).flatten()    # Create a column of ones
X = np.column_stack([ones, initialX])      # Add the column to X
theta = np.zeros((2, 1))            # Initialise fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function...\n')
# Compute and display initial cost

J = computeCost(X, y, theta)

print('With theta = [[0], [0]]\nCost computed = ' + str(J))
print('Expected cost value (approx) 32.07\n')

# Further testing of cost function
theta = np.array([[-1], [2]])
J = computeCost(X, y, theta)

print('With theta = [[-1], [2]]\nCost computed = ' + str(J))
print('Expected cost value (approx) 54.24\n')

input('Program paused. Press enter to continue.\n')

print('\nRunning Gradient Descent ...\n')

# run gradient descent
theta = np.zeros((2, 1)) 
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print(str(theta.flat[0]))
print(str(theta.flat[1]) + '\n')

print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.scatter(initialX, y, marker='x', label='Training data', color='red')
plt.plot(initialX, X.dot(theta), label='Linear regression')
plt.legend()
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of ' + str(predict1 * 10000) + '\n')
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of ' + str(predict2 * 10000) + '\n')

input('Program paused. Press enter to continue.\n')

# Part 4: Visualizing J(theta_0, theta_1)

print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i][j] = computeCost(X, y, t);


# Surface plot
xAxis, yAxis = np.meshgrid(theta0_vals, theta1_vals)
zAxis = np.transpose(J_vals)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xAxis, yAxis, zAxis)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()


