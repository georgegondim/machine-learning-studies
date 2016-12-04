import numpy as np
import matplotlib.pyplot as plt

from knn import KNN

def get_data():
    width = 8
    height = 8
    N = width * height
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    start_t = 0
    n = 0
    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            Y[n] = t
            t = (t + 1) % 2
            n += 1
        start_t = (start_t + 1) % 2
    
    return X, Y
    

if __name__ == '__main__':
    X, Y = get_data()
    
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()
   
    for k in (1, 2, 3):
        model = KNN(k)
        model.fit(X, Y)
        print('k', k, 'Train accuracy: ', model.score(X, Y))    
    