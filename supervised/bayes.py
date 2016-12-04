import numpy as np
from scipy.stats import multivariate_normal as mvn
from util import get_data_mnist
from time import time

class Bayes(object):
    def fit(self, X, Y, smoothing=10e-3):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            Xc = X[Y == c]
            self.gaussians[c] = {
                'mean': Xc.mean(axis=0),
                'cov': np.cov(Xc.T) + smoothing * np.eye(D)
            }
            self.priors[c] = np.log(float(len(Xc)) / len(X))
    
    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, cov = g['mean'], g['cov']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=cov) + self.priors[c]
        
        return np.argmax(P, axis=1)
   
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
        
if __name__ == '__main__':
    X, Y = get_data_mnist(10000)
    Ntrain = len(X) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    
    
    model = Bayes()
    
    t0 = time()
    model.fit(X, Y)
    print('Fit time: %.3fs' % (time() - t0))
    
    t0 = time()
    Strain = model.score(Xtrain, Ytrain)
    t1 = time()
    Stest = model.score(Xtest, Ytest)
    t2 = time()
    
    print('Train: acc= %.3f, time= %.3f' %(Strain, t1 - t0))
    print('Test: acc= %.3f, time= %.3f' %(Stest, t2 - t1))