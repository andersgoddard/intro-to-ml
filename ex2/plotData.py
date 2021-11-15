import matplotlib.pyplot as plt

def plotData(X, y, plot=True):
    # Plots the data points X and y into a new figure 
    # PLOTDATA(x,y) plots the data points with + for the positive examples
    # and o for the negative examples. X is assumed to be a Mx2 matrix
    
    pos = X[y==1]
    neg = X[y!=1]
    
    plt.scatter(pos[0], pos[1], marker='x', label='Admitted')
    plt.scatter(neg[0], neg[1], marker='o', label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    
    if plot:
        plt.show()
