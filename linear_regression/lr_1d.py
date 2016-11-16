import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data_1d.csv').as_matrix()
X = data[:, 0]
Y = data[:, 1]

# Compute intermediates values
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

print 'Rsquared = ', Rsquared