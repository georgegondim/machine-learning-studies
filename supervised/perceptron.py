import numpy as np
from util import get_data_mnist, get_data_xor
from time import time
import matplotlib.pyplot as plt

def get_data():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300, 2)) * 2 - 1
    Y = np.sign(X.dot(w) + b)
    
    return X, Y;

class Perceptron(object):    
    def fit(self, X, Y, lr=1.0, nepochs=1000):
        self.w = np.random.randn(X.shape[1])
        self.w[0] = 0
        
        costs = []
        for i in range(nepochs):
            P = self.predict(X)
            errors = np.nonzero(P != Y)[0]
            if len(errors) == 0:
                break;
                
            self.w += (lr * X.T * Y.T).T.sum(axis=0)
            c = len(errors) / float(X.shape[0])
            costs.append(c)
            if abs(c - sum(costs[-5:]) / 5.0)  < 0.000001:
                break;
                
        plt.plot(costs)
        plt.show()
        
    def predict(self, X):
        return np.sign(X.dot(self.w))
        
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

if __name__ == '__main__':  
    def test(X, Y, lr, name):
        Ntrain = int(len(X) / 2)
        Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
        Xtest, Ytest = X[Ntrain:, :], Y[Ntrain:]
        
        model = Perceptron()
        t0 = time()
        model.fit(Xtrain, Ytrain, lr=lr, nepochs=1000)
        print('%s Fit time: %.3fs' % (name, time() - t0))
        
        t0 = time()
        Strain = model.score(Xtrain, Ytrain)
        t1 = time()
        Stest = model.score(Xtest, Ytest)
        t2 = time()
        
        print('%s Train: acc= %f, time= %.3fs' %(name, Strain, t1 - t0))
        print('%s Test: acc= %f, time= %.3fs' %(name, Stest, t2 - t1))
        
        
    # Linear data
    X, Y = get_data() 
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()
    test(X, Y, 1, 'Linear')
    
    # MNIST data
    pos = 1
    X, Y = get_data_mnist()
    Y[Y == pos] = 1
    Y[Y != pos] = -1
    
    test(X, Y, 10e-3, 'MNIST')

    # XOR data
    X, Y = get_data_xor() 
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()
    test(X, Y, 10e-3, 'XOR')