# -*- coding: utf-8 -*-
import numpy as np
from ecommerce_preprocess import load_binary_data

X, T = load_binary_data()
N, D = X.shape
X = np.concatenate((np.array([[1]*N]).T, X), axis = 1)
W = np.random.randn(D + 1)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))
    
def forward(X, W):
    return logistic(X.dot(W))
    
def accuracy(T, Y):
    return np.mean(T == np.round(Y))

def cross_entropy(T, Y):
    return -np.nansum(T * np.log(Y) + (1 - T)  * np.log(1 - Y))
 
Y = forward(X,W)
print "Accuracy: ", accuracy(T, Y)
print "Cross-Entropy: " , cross_entropy(T, Y)