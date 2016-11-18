import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    
    # Plot data and predicted line
    tmp = np.concatenate((np.array([X[:, 1]]).T, Ypred), axis=1)
    print tmp
    tmp = tmp[np.argsort(tmp[:,0])]
    plt.scatter(X[:, 1], Y, label='Data')
    lines = plt.plot(tmp[:, 0], tmp[:, 1], label='Regression')
    plt.setp(lines, color='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    
    print "Model y = a + b*x + c*x^2 -- ", W.T
    print 'Rsquared = ', Rsquared   
    
def load_data():
    data = pd.read_csv('data_poly.csv').as_matrix()
    data = scale(data, axis=0)
    X = np.ones([data.shape[0], 3]);
    X[:, 1] = data[:, 0]
    X[:, 2] = X[:, 1] * X[:, 1]
    Y = data[:, 1]
    Y.shape = (Y.shape[0], 1)
    return X, Y
    
if __name__ == "__main__":
    main()