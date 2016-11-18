import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

def main():
    X,Y = load_data()
    regression(X, Y)

def regression(X, Y):
    W = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Ypred = X.dot(W)
    residues = Y - Ypred
    SSresidues = residues.T.dot(residues).sum()
    deviations = Y - Y.mean()
    SSdeviations = deviations.T.dot(deviations).sum()
    Rsquared = 1 - SSresidues / SSdeviations
    
    print 'Model y = a + b*x1 + c*x2', W[1, 0], W[1, 0], W[2, 0]
    print 'Rsquared = ', Rsquared   
    
def load_data():
    data = pd.read_csv('data_2d.csv').as_matrix()
    data = scale(data, axis=0)
    X = np.ones(data.shape);
    X[:, 1:X.shape[1]] = data[:, 0:-1]
    Y = data[:, -1]
    Y.shape = (Y.shape[0], 1)
    return X, Y
    
if __name__ == "__main__":
    main()