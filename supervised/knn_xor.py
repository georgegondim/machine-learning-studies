import matplotlib.pyplot as plt
from knn import KNN
from util import get_data_xor

if __name__ == '__main__':
    X, Y = get_data_xor()
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    
    for k in (1, 2, 3, 4, 5):
        model = KNN(k)
        model.fit(X, Y)
        print('k', k, 'Train accuracy: ', model.score(X, Y))    
    
