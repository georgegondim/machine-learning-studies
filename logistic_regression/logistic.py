# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
from logistic_regression import logistic_regression

N = 100
D = 2

X = np.random.randn(N, D)
X[:ceil(N/2), :] = X[:ceil(N/2), :]  - 2
X[floor(N/2):, :] = X[floor(N/2):, :]  + 2

T = np.array([0]*ceil(N/2) + [1]*floor(N/2))

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)
w = np.random.rand(D + 1)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(X, W):
    return logistic(X.dot(W))

def cross_entropy(T, Y):
    return -np.nansum(T * np.log(Y) + (1 - T)  * np.log(1 - Y))

def accuracy(T, Y):
    return np.mean(T == np.round(Y))

Y =  forward(Xb, w)

print("Random weights:")
print("\tAccuracy: ", accuracy(T, Y))
print("\tCross-Entropy: " , cross_entropy(T, Y))

plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()

logistic_regression(Xb, T, 5000, 0.0001, 0)