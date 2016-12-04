import numpy as np
from scipy import stats
from util import get_data_mnist
import time

class KNN(object):
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            d = np.linalg.norm(self.X - x, axis=1)
            indexes = d.argsort()[0:self.k]
            y[i] = stats.mode(self.y[indexes])[0]
        return y.astype(np.int32)
      
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
        
if __name__ == '__main__':
    limit = 2000
    N_train = int(0.5*limit)
    X, Y = get_data_mnist(limit)
    N, D = X.shape
    for k in (1,2,3,4,5):
        knn = KNN(k)
        knn.fit(X[0:N_train, :], Y[0:N_train])
        t0 = time.time()
        score = knn.score(X[N_train:, :], Y[N_train:])
        print ("k: %d, elapsed time: %.3fs, score: %.3f" % 
               (k, time.time() - t0, score))
        