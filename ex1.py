import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

print('Running warmUpExercise ...\n')
print('5 x 5 Identity Matrix: \n')

fiveByFive = np.identity(5)
              
print(fiveByFive)

input('Program paused. Press enter to continue.\n')

data = pd.read_csv('ex1data1.csv')

X = data.iloc[:, 0]
y = data.iloc[:, 1]
m = len(y)

plt.scatter(X, y, marker='x')
plt.show()

input('Program paused. Press enter to continue.\n')