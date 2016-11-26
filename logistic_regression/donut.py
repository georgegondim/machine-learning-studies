import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
from logistic_regression import logistic_regression

N = 1000
D = 2

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
T = np.array([0]*N_inner + [1]*N_outer)
T.shape = (T.shape[0], 1)

plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
plt.show()

ones = np.ones([N, 1])
r = np.sqrt((X * X).sum(axis=1, keepdims=True))
Xb = np.concatenate((ones, r, X), axis=1)
logistic_regression(Xb, T, 5000, 0.0001, 0)