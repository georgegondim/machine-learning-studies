# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

def load_data():
    data = pd.read_csv('ecommerce_data.csv').as_matrix()
    X = data[:, 0:-1]
    Y = data[:, -1]
    X[:, [1, 2]] = scale(X[:,[1, 2]])
    
    N, D = X.shape
    
    # one-hot
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    X2[:,-4:] = Z
    
    return X2, Y

def load_binary_data():
    X, Y = load_data()
    X = X[Y <= 1.0]
    Y = Y[Y <= 1.0]
    return X, Y
    
if __name__ == "__main__":
    X, Y = load_binary_data()