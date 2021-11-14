import numpy as np
from computeCost import *

def gradientDescent(X, y, theta, alpha, num_iters):
    # Performs gradient descent to learn theta
    # updating theta by taking num_iters gradient steps 
    # with learning rate alpha

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
      
        #  ====================== YOUR CODE HERE ======================
        #  Instructions: Perform a single gradient step on the parameter vector
        #                theta. 
        # 
        #  Hint: While debugging, it can be useful to print out the values
        #        of the cost function (computeCost) and gradient here.
        # 

        hX = X.dot(theta).flatten()
        error = hX - y
        theta = theta.flatten() - ((alpha/m) * np.transpose(X).dot(error))

        #  ============================================================

        # #  Save the cost J in every iteration    
        # J_history[i] = computeCost(X, y, theta);
        
    return theta
