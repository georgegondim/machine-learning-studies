import pandas as pd
import numpy as np

def load_data(limit=None):
    data = pd.read_csv('../datasets/mnist/train.csv').as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y
