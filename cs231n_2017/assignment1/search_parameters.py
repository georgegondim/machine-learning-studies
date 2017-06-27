import numpy as np
from cs231n.classifiers.neural_net import TwoLayerNet
import time
from cs231n.data_utils import load_CIFAR10
import joblib
from joblib import Parallel, delayed


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

input_size = 32 * 32 * 3
num_classes = 10

def get_modelname(acc, params):
    modelname = 'model.acc:%f.lr:%f.reg:%f.dropout.%f' % \
        (acc, params[0], params[1], params[2])
    return modelname

def train(params):
    print('\tevaluating %s' % str(params))
    log_lr = params[0]
    log_rs = params[1]
    lr_decay = params[2]

    # Train the network
    lr = np.power(10, log_lr)
    rs = np.power(10, log_rs)

    tic = time.time()
    net = TwoLayerNet(input_size, hidden_size, num_classes, p_dropout=p_dropout)
    stat = net.train(X_train, y_train, X_val, y_val,
                     num_iters=num_iters, batch_size=200,
                     learning_rate=lr, learning_rate_decay=lr_decay,
                     reg=rs, verbose=verbose, optimizer='adam')
    validation_accuracy = max(stat['val_acc_history'])
    modelname = get_modelname(validation_accuracy, params)
    joblib.dump(stat, './checkpoints/' + modelname + '.pkl')
    net.save_model('./checkpoints/' + modelname)
    log = '\taccuracy=%.6f, params=%s, took=%f' % \
        (validation_accuracy, str(params), time.time() - tic)
    print(log)
    fp.write(log + "\n")
    fp.flush()

# Limits
intervals = {}
intervals['log_lr'] = [-3.8, -3.5]
intervals['log_rs'] = [-1.5, -1.2]
intervals['lr_decay'] = [0.89, 0.96]
#intervals['dropout'] = 0.01*np.arange(20, 45, 5)
#intervals['hidden_size'] = np.arange(1200, 1600, 100)

# Search parameters
num_runs = 300
params_list = []
for run in range(num_runs):
    log_lr = np.random.uniform(intervals['log_lr'][0], intervals['log_lr'][1])
    log_rs = np.random.uniform(intervals['log_rs'][0], intervals['log_rs'][1])
    lr_decay = np.random.uniform(intervals['lr_decay'][0], intervals['lr_decay'][1])
    #hidden_size = intervals['hidden_size'][np.random.randint(intervals['hidden_size'].size)]
    #p_dropout = intervals['dropout'][np.random.randint(intervals['dropout'].size)]
    params_list.append([log_lr, log_rs, lr_decay])

# Execution parameters
verbose = False
hidden_size = 1500
num_iters = 10000
stats = []
p_dropout = 0.3

num_workers = 1
fp = open("results3.log", "a")

for i, param in enumerate(params_list):
    print('Run %d / %d' % (i,  num_runs))
    train(param)
