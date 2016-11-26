import numpy as np
import matplotlib.pyplot as plt

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(X, W):
    return logistic(X.dot(W))

def cross_entropy(T, Y):
    return -np.nansum(T*np.log(Y) + (1 - T)*np.log(1-Y))

def accuracy(T, Y):
    return np.mean(T == np.round(Y))

def logistic_regression(X, T, nepochs, lr, lmbda):
    T.shape = (T.shape[0], 1)
    w = np.random.randn(X.shape[1], 1)    
    error = []
    Y = forward(X, w)
    for i in range(nepochs):
        e = cross_entropy(T, Y)
        error.append(e)
        if i%100 == 0:
            print(e)
        w += -lr * X.T.dot(Y - T) - lmbda * w
        Y = forward(X, w)
    
    print('Final w: ', w.T)
    print('Final accuracy: ', accuracy(T, Y))
    
    plt.plot(error)
    plt.show()