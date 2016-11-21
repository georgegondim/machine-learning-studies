# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)
X[:N/2, :] = X[:N/2, :]  - 2
X[N/2:, :] = X[N/2:, :]  + 2

T = np.array([0]*(N/2) + [1]*(N/2))

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
    
print "Random weights:"
print "\tAccuracy: ", accuracy(T, Y)
print "\tCross-Entropy: " , cross_entropy(T, Y)

plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()

lr = 0.1
for i in xrange(100):
    if i%10 == 0:
        print "GD loss: ", cross_entropy(T, Y)
    w -= lr*Xb.T.dot(Y - T)
    Y = forward(Xb, w)
print 'Final w: ', w.T
print 'Final loss: ', cross_entropy(T, Y)
print 'Final accuracy: ', accuracy(T, Y)