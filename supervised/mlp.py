import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from util import get_data_mnist
plt.style.use('ggplot')
import time

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

def batch_iteration(W1, W2, X, y, l2_coef):
    # Add biases to hidden layer
    inputs = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)

    # Compute hidden layer activation
    hidden_inputs = W1.dot(inputs.T)
    hidden_activations = sigmoid(hidden_inputs)

    # Add biases to output
    hidden_activations = np.concatenate([np.ones([1, hidden_activations.shape[1]]), hidden_activations], axis=0)

    # Compute outputs
    output_inputs = W2.dot(hidden_activations)
    outputs = sigmoid(output_inputs).T

    # One hot encode label vector
    N_samples = outputs.shape[0]
    y = y.astype(int)
    y.shape = [y.shape[0], 1]
    idx_y = np.arange(y.shape[0]).reshape(y.shape[0], 1)
    Y = np.zeros([y.shape[0], outputs.shape[1]])
    Y[idx_y, y] = 1

    # Compute loss function: cross-entropy
    J = (-Y * np.log(outputs) - (1 - Y) * np.log(1 - outputs)).sum() / N_samples

    # Compute L2 regularization factor
    W1_reg = W1[:, 1:].flatten()
    W2_reg = W2[:, 1:].flatten()
    J += l2_coef * (W1_reg.dot(W1_reg) + W2_reg.dot(W2_reg)) / 2 / N_samples

    # Compute gradients with backpropagation
    W2_errors = (outputs - Y).T
    W1_errors = (W2[:, 1:]).T.dot(W2_errors) * sigmoid_derivative(hidden_inputs)

    W1_grads = W1_errors.dot(inputs) / N_samples
    W2_grads = W2_errors.dot(hidden_activations.T) / N_samples

    W1_grads[:, 1:] = W1_grads[:, 1:] + l2_coef * W1[:, 1:] / N_samples
    W2_grads[:, 1:] = W2_grads[:, 1:] + l2_coef * W2[:, 1:] / N_samples

    return outputs, J, W1_grads, W2_grads

def grad_check():
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;
    l2_coef = 0.001

    W1 = 0.3*np.random.randn(hidden_layer_size, input_layer_size + 1)
    W2 = 0.3*np.random.randn(num_labels, hidden_layer_size + 1)
    X = 0.3*np.random.randn(m, input_layer_size)
    y  = np.mod(range(m), num_labels).T;

    cost = lambda W1, W2 : batch_iteration(W1, W2, X, y, l2_coef)
    outputs, J, W1_grads, W2_grads = cost(W1, W2)

    # Compute Numerical Gradients
    nparams = W1.size + W2.size
    numgrad = np.zeros(nparams);

    W1_perturb = np.zeros(W1.shape)
    W2_perturb = np.zeros(W2.shape)
    e = 1e-4;
    for p in range(nparams):
        if p < W1.size:
            row = int(p / W1.shape[1])
            col = p % W1.shape[1]
            W1_perturb[row, col] = e
        else:
            row = int((p - W1.size) / W2.shape[1])
            col = (p - W1.size) % W2.shape[1]
            W2_perturb[row, col] = e

        # Set perturbation vector
        _, loss1, _, _ = cost(W1 - W1_perturb, W2 - W2_perturb);
        _, loss2, _, _ = cost(W1 + W1_perturb, W2 + W2_perturb);

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e);

        if p < W1.size:
            W1_perturb[row, col] = 0
        else:
            W2_perturb[row, col] = 0

    grads = np.concatenate([W1_grads.flatten(), W2_grads.flatten()], axis=0)
    print("numgrads - grads = ", numgrad - grads)

def accuracy(output, y):
    pred = np.argmax(output, axis=1)
    return (pred == y).mean()

def train_mnist():
    # Data
    print('Loading data...')
    X, Y = get_data_mnist()
    idx = int(0.8*X.shape[0])
    X_train = X[:idx, :]
    Y_train = Y[:idx]
    X_val = X[idx:, :]
    Y_val = Y[idx:]
    X = []
    Y = []

    # Parameters
    print('Initializing model...')
    input_layer_size = X_train.shape[1]
    hidden_layer_size = 100
    output_layer_size = 10
    l2_coef = 0.01

    # Weights initialization
    W1 = 0.3*np.random.randn(hidden_layer_size, input_layer_size + 1)
    W2 = 0.3*np.random.randn(output_layer_size, hidden_layer_size + 1)

    niter = 2000
    hist_J = np.zeros(niter)
    hist_J_val = np.zeros(niter)
    learning_rate = 0.7
    for i in range(niter):
        t0 = time.time()
        outputs, hist_J[i], W1_grads, W2_grads = batch_iteration(W1, W2, X_train, Y_train, l2_coef)
        outputs_val, hist_J_val[i], _, _ = batch_iteration(W1, W2, X_val, Y_val, l2_coef)

        W1 = W1 - learning_rate * W1_grads
        W2 = W2 - learning_rate * W2_grads

        print ("Epoch: %d - Elapsed time: %.4fs\n\tTraining Loss = %.4f, Training ACC = %.4f\n\tValidation Loss = %.4f, Validation ACC = %.4f\n\tGradient mean norm = %e"
        % (i, time.time() - t0, hist_J[i], accuracy(outputs, Y_train), hist_J_val[i], accuracy(outputs_val, Y_val), (np.abs(W1_grads).sum() + np.abs(W2_grads).sum()) / (W1_grads.size + W2_grads.size)))

    plt.plot(hist_J, alpha=0.5, label='Training')
    plt.plot(hist_J_val, alpha=0.5, label='Validation')
    plt.title('Loss')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    train_mnist()
