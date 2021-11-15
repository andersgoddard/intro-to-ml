import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotData import *

def plotDecisionBoundary(theta, X, y, a):
    X_plot = pd.DataFrame(X[:, 1:3])    
    plotData(X_plot, y, False) 
    
    if np.shape(X)[1] <= 3:
        #  Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X[0][:])-2, max(X[0][:])+2]
        plot_x = np.array(plot_x)

        #  Calculate the decision boundary line
        plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])

        #  Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)
        
        #  Legend, specific for the exercise
        plt.axis([30, 100, 30, 100])
        plt.show()

    else
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)));
        
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = mapFeature(u[i], v[j]).dot(theta)
        
        # important to transpose z before calling contour
        z = np.transpose(z)

        # Plot z = 0
        
        # Coming back to this...

