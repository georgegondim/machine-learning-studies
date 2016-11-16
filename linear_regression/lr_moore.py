import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def main():
    # Load data
    X = []
    Y = []
    
    non_decimal = re.compile(r'[^\d]+')
    
    for line in open('moore.csv'):
        r = line.split('\t')
        x = int(non_decimal.sub('', r[2].split('[')[0]))
        y = int(non_decimal.sub('', r[1].split('[')[0]))
        X.append(x)
        Y.append(y)
        
    X = np.array(X)
    Y = np.array(Y)
    plt.plot(X, Y)
    plt.show()
    
    Ylog = np.log(Y)
    regression(X, Ylog)
    
def regression(X, Y):
    # Compute intermediate values
    Ymean = Y.mean()
    Xmean = X.mean()
    Xsum = X.sum()
    Xsquared = X.dot(X)
    XYsum = X.dot(Y)
    denominator = Xsquared - Xmean*Xsum
    
    # Compute a and b from y = ax + b
    a = (XYsum - Ymean*Xsum) / denominator
    b = (Ymean*Xsquared - Xmean*XYsum) / denominator

    # Compute predicted values
    Ypred = a*X + b
    
    # Plot data and predicted line
    plt.scatter(X, Y, label='Data')
    lines = plt.plot(X, Ypred, label='Regression')
    plt.setp(lines, color='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    
    # Compute Rsquared
    residues = Y - Ypred
    SSresidues = residues.dot(residues)
    deviations = Y - Ymean
    SSdeviations = deviations.dot(deviations)
    Rsquared = 1 - SSresidues / SSdeviations
    print 'a = ', a, ', b = ', b    
    print 'Rsquared = ', Rsquared
    
    # Time to double
    time_to_double = np.log(2) / a
    print 'Time to double = ', time_to_double, ' years' 
    
    
if __name__ == "__main__":
    main()