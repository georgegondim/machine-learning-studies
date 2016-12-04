import pandas as pd
import numpy as np
from math import ceil, floor

def get_data_mnist(limit=None):
    data = pd.read_csv('../datasets/mnist/train.csv').as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

def get_data_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5
    X[50:100] = np.random.random((50, 2)) / 2
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])
    Y = np.array([0]*100 + [1]*100)
    return X, Y
    
def get_data_donut():
    N = 200
    R_inner = 5
    R_outer = 10
    
    N_inner = int(ceil(N/2))
    N_outer = int(floor(N/2))

    R1 = np.random.randn(N_inner) + R_inner
    theta = 2 * np.pi * np.random.random(N_inner)
    X_inner = np.concatenate([[R1 * np.cos(theta)], 
                          [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N_outer) + R_outer
    theta = 2 * np.pi * np.random.random(N_outer)
    X_outer = np.concatenate([[R2 * np.cos(theta)], 
                              [R2 * np.sin(theta)]]).T
    
    X = np.concatenate([X_inner, X_outer], axis=0)
    Y = np.array([0]*N_inner + [1]*N_outer)

    return X, Y
