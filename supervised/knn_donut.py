import matplotlib.pyplot as plt
from util import get_data_donut
from knn import KNN

if __name__ == '__main__':
    X, Y = get_data_donut()
    
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()
   
    for k in (1, 2, 3, 4, 5):
        model = KNN(k)
        model.fit(X, Y)
        print('k', k, 'Train accuracy: ', model.score(X, Y))    