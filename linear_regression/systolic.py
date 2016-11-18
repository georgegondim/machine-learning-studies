# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

def main():
    X, Y = load_data()
    plt.scatter(X[:, 1], Y)
    plt.show()
    plt.scatter(X[:, 2], Y)
    plt.show()
    
    print '!!! X2 only'
    X2 = np.array(X[:, [0, 1]])
    regression(X2, Y)
    
    print '!!! X3 only'
    X3 = np.array(X[:, [0, 2]])
    regression(X3, Y)
    
    print '!!! X1 and X2'
    regression(X, Y)
    
    print '!!! X1, X2 and random'
    rand_col = np.random.randn(1, X.shape[0]).T
    Xr = np.concatenate([X, rand_col], axis=1)
    Xr[:, 0:3] = X;
    regression(Xr, Y)
    
def regression(X, Y):
    W = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Ypred = X.dot(W)
    residues = Y - Ypred
    SSresidues = residues.T.dot(residues).sum()
    deviations = Y - Y.mean()
    SSdeviations = deviations.T.dot(deviations).sum()
    Rsquared = 1 - SSresidues / SSdeviations
    
    if X.shape[1] == 2:
        plt.scatter(X[:, 1], Y)
        lines = plt.plot(sorted(X[:, 1]), sorted(Ypred), label='Regression')
        plt.setp(lines, color='r')
        plt.show()
    print "Weights = ", W.T
    print 'Rsquared = ', Rsquared       
                
def load_data():
    data = pd.read_excel('mlr02.xls', header=None).as_matrix()[1:, :].astype(float)
    data = scale(data, axis=0)
    X = np.ones(data.shape);
    X[:, [1, 2]] = data[:, [1, 2]]
    Y = data[:, 0]
    Y.shape = (Y.shape[0], 1)
    return X, Y
    
if __name__ == "__main__":
    main()