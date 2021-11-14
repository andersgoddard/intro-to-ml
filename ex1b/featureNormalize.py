import numpy as np

def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1, np.shape(X)[1]))
    sigma = np.zeros((1, np.shape(X)[1]))
    
    mu = np.mean(X)
    sigma = np.std(X)
    
    print('mu: ' + str(mu))
    print('sigma: ' + str(sigma))
    
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma