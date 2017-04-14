import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras

##############################################
# Preprocess data with keras

# Input image dimensions
img_rows, img_cols = 28, 28
img_pixels = img_rows * img_cols

# Network dimensions
input_size = img_pixels
hidden_size = 100
output_size = 10

# The data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train.reshape(x_train.shape[0], img_pixels)
x_test = x_test.reshape(x_test.shape[0], img_pixels)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# One hot encode labels
y_train = keras.utils.to_categorical(y_train, output_size)
y_test = keras.utils.to_categorical(y_test, output_size)

# hold out validation data
idx = int((5 / 6) * x_train.shape[0])
x_val = x_train[idx:, :]
y_val = y_train[idx:, :]
x_train = x_train[:idx, :]
y_train = y_train[:idx, :]
##############################################

# hyperparameters
learning_rate = 0.1
training_epochs = 200
batch_size = 200
momentum = 0.99

# Store validation and test sets
tf_x_val = tf.constant(x_val)
tf_x_test = tf.constant(x_test)


def inference(x):
    # Weights and bias initialization
    hidden_weights = tf.Variable(tf.random_normal(
        [hidden_size, input_size], stddev=0.3))
    hidden_bias = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.3))
    output_weights = tf.Variable(tf.random_normal(
        [output_size, hidden_size], stddev=0.3))
    output_bias = tf.Variable(tf.random_normal(
        [output_size, 1], stddev=0.3))

    # Feedforward
    hidden_activations = tf.nn.sigmoid(
        tf.matmul(hidden_weights, tf.transpose(x)) + hidden_bias)
    output_activations = tf.transpose(tf.nn.softmax(
        tf.matmul(output_weights, hidden_activations) + output_bias))

    return output_activations


def loss(outputs, y):
    # Cross-entropy loss
    cross_entropy = -tf.reduce_sum(y * tf.log(outputs), reduction_indices=1)
    return tf.reduce_mean(cross_entropy)


def get_corrects(outputs, y):
    prediction = tf.argmax(outputs, axis=1)
    correct = tf.argmax(y, axis=1)
    return tf.reduce_sum(tf.cast(tf.equal(prediction, correct), tf.int32))


def evaluate(sess, loss, correct_count, x_placeholder, y_placeholder, x, y, optimizer=None):
    corrects = 0
    avg_loss = 0.0
    total_batches = int(x.shape[0] / batch_size)

    for batch in range(total_batches):
        idx_lower = batch * batch_size
        idx_upper = min(idx_lower + batch_size, x.shape[0])
        batch_x = x[idx_lower:idx_upper, :]
        batch_y = y[idx_lower:idx_upper, :]

        if optimizer is None:
            l, c = sess.run(
                [loss, correct_count],
                feed_dict={x_placeholder: batch_x, y_placeholder: batch_y})
        else:
            _, l, c = sess.run(
                [optimizer, loss, correct_count],
                feed_dict={x_placeholder: batch_x, y_placeholder: batch_y})

        corrects += c
        avg_loss += l / total_batches

    return avg_loss, float(corrects) / x.shape[0]


# Placeholders to inputs and outputs
x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

# Get outputs and loss
outputs = inference(x)
loss = loss(outputs, y)

# Count correct predictions
correct_count = get_corrects(outputs, y)

# SGD with nesterov momentum optimizer
optimizer = tf.train.MomentumOptimizer(
    learning_rate, momentum, use_nesterov=True).minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()

# Number of batches
total_batches = int(x_train.shape[0] / batch_size)

# Start graph
with tf.Session() as sess:
    sess.run(init)

    # Training
    for epoch in range(training_epochs):
        train_loss, train_accuracy = evaluate(
            sess, loss, correct_count, x, y, x_train, y_train, optimizer)
        val_loss, val_accuracy = evaluate(
            sess, loss, correct_count, x, y, x_val, y_val)

        # Print log
        print('Epoch: %04d\n\ttrain_loss=%.9f, train_accuracy=%.5f'
              '\n\tval_loss=%.9f, val_accuracy=%.5f' % (epoch, train_loss, train_accuracy, val_loss, val_accuracy))
    test_loss, test_accuracy = evaluate(
        sess, loss, correct_count, x, y, x_test, y_test)
    print('\ntest_loss=%.9f, test_accuracy=%.5f' %
          (test_loss, test_accuracy))
