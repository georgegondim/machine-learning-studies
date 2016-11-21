# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from ecommerce_preprocess import load_binary_data

X, Y = load_binary_data()
X, Y = shuffle(X, Y)
N, D = X.shape

X = np.concatenate((np.array([[1]*N]).T, X), axis = 1)
Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

W = np.random.randn(D + 1)

def logistic(z):
    return 1 / (1 + np.exp(-z))
    
def forward(X, W):
    return logistic(X.dot(W))
    
def accuracy(T, Y):
    return np.mean(T == np.round(Y))

def cross_entropy(T, pY):
    return -np.mean(T * np.log(pY) + (1 - T)  * np.log(1 - pY))
 
train_costs = []
test_costs = []
lr = 0.001
for i in xrange(10000):
    pYtrain = forward(Xtrain, W)
    pYtest = forward(Xtest, W)
    ctrain = cross_entropy(Ytrain, pYtrain);
    ctest = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)
    W -= lr*Xtrain.T.dot(pYtrain - Ytrain)
    if i % 1000 == 0:
        print "It: ", i, "Train: ", ctrain, "Test: ", ctest
    
print "Train Accuracy", accuracy(Ytrain, pYtrain)
print "Test Accuracy", accuracy(Ytest, pYtest)

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
    