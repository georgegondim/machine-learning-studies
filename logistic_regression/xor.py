import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import logistic_regression

N = 4
D = 2

X = np.array([[0 , 0], [0 , 1], [1 , 0], [1 , 1]])
T = np.array([0, 1, 1, 0])
T.shape = (4, 1)
                
plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
plt.show()

ones = np.array([[1]*N]).T

xy = np.array(np.matrix(X[:, 0] * X[:,1]).T)
Xb = np.concatenate((ones, xy, X), axis=1)

logistic_regression(Xb, T, 5000, 0.01, 0)