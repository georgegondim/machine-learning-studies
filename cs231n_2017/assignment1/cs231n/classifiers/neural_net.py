from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
import time


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 p_dropout=0., std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}

        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.p_keep = 1 - p_dropout

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # First layer
        z1 = X.dot(W1) + b1

        # ReLU
        a1 = z1
        negative_a1 = a1 < 0.
        a1[negative_a1] = 0.

        # Dropout
        dropout = (np.random.rand(*a1.shape) < self.p_keep) / self.p_keep
        a1 *= dropout

        # Second layer
        scores = a1.dot(W2) + b2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        num_classes = W2.shape[1]

        # One hot encode y
        y_one_hot = np.zeros((N, num_classes))
        y_one_hot[np.arange(N), y] = 1

        # Compute probabilities
        tmp = scores - np.amax(scores, axis=-1, keepdims=True)
        unnorm_prob = np.exp(tmp)
        a2 = unnorm_prob / np.sum(unnorm_prob, axis=-1, keepdims=True)

        # Compute cross-entropy + L2 regularization
        loss = -np.sum(y_one_hot * np.log(a2 + 1e-8)) / N + \
            reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        error2 = a2 - y_one_hot
        dW2 = a1.transpose().dot(error2) / N + reg * 2 * W2
        db2 = np.sum(error2, axis=0) / N

        error1 = error2.dot(W2.transpose()) * (1 - negative_a1) * dropout
        dW1 = X.transpose().dot(error1) / N + reg * 2 * W1
        db1 = np.sum(error1, axis=0) / N

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False,
              optimizer='sgd', mu=0.9, early_stop=False, filename=None):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        theta = np.zeros((0,))
        for param in self.params:
            theta = np.concatenate((theta, self.params[param].flatten()))
        nparams = theta.size

        # Adam momentums and parameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = np.zeros((nparams,))
        v = np.zeros((nparams,))

        best_acc = -1
        count_stop = 0
        for it in xrange(num_iters):
            tic = time.time()
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            batch_idx = np.random.choice(num_train, size=batch_size)
            X_batch = X[batch_idx, :]
            y_batch = y[batch_idx]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            dtheta = np.zeros((0,))
            for param in self.params:
                dtheta = np.concatenate((dtheta, grads[param].flatten()))

            if optimizer == 'adam':  # Adam
                m = beta1 * m + (1 - beta1) * dtheta
                mt = m / (1 - beta1 ** (it + 1))
                v = beta2 * v + (1 - beta2) * (dtheta ** 2)
                vt = v / (1 - beta2 ** (it + 1))
                theta += -learning_rate * mt / (np.sqrt(vt) + eps)
            elif optimizer == 'nesterov':  # SGD with Nesterov Momentum
                v_prev = v
                v = mu * v - learning_rate * dtheta
                theta += -mu * v_prev + (1 + mu) * v
            else:  # SGD
                theta -= learning_rate * dtheta

            idx = 0
            for param in self.params:
                size = self.params[param].size
                self.params[param] = np.reshape(theta[idx:idx + size],
                                                self.params[param].shape)
                idx += size

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                    count_stop = 0
                    if filename is not None:
                        self.save_model(filename)
                else:
                    count_stop += 1
                    if early_stop and count_stop == 5:
                        break

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # First layer
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, z1)

        # Second layer
        z2 = a1.dot(self.params['W2']) + self.params['b2']
        scores = z2

        y_pred = np.argmax(scores, axis=-1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred

    def save_model(self, filename):
        """
        Save model parameters to file
            Inputs:
            - filename: string with file name
        """
        arrs = []
        for param in self.params:
            arrs.append(self.params[param])
        np.savez(filename, arrs)

    def load_model(self, filename):
        """
        Load model parameters from file
            Inputs:
            - filename: string with file name
        """
        arrs = np.load(filename)['arr_0']
        for i, param in enumerate(self.params):
            self.params[param] = arrs[i]
