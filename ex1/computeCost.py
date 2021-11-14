def computeCost(X, y, theta):
    # Compute cost for linear regression using theta as the
    # parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y)  # number of training examples
    J = 0       # return value

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    Xtheta = X.dot(theta).flatten()
    z = Xtheta - y

    J = (1 / (2 * m)) * sum(pow(z, 2))
    
    return J