'''
 Implemeting a similar model from the paper:
    Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. 2003. A neural probabilistic language model. J. Mach. Learn. Res. 3 (March 2003), 1137-1155.
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, Flatten, Reshape
from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.optimizers import SGD, RMSprop
from scipy.io import loadmat
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# Load and preprocess dataset from Hinton's Course on Coursera
input_length = 3
data = loadmat('data.mat')
x_test = data['data'][0][0][0].T[:, :input_length] - 1
x_train = data['data'][0][0][1].T[:, :input_length] - 1
x_valid = data['data'][0][0][2].T[:, :input_length] - 1
y_test = data['data'][0][0][0].T[:, input_length] - 1
y_train = data['data'][0][0][1].T[:, input_length] - 1
y_valid = data['data'][0][0][2].T[:, input_length] - 1
y_test = to_categorical(y_test.reshape(y_test.shape[0], 1))
y_train = to_categorical(y_train.reshape(y_train.shape[0], 1))
y_valid = to_categorical(y_valid.reshape(y_valid.shape[0], 1))
vocab = data['data'][0][0][3].T

# Hyperparameters
batchsize = 100  # Mini-batch size.; default = 100
learning_rate = 0.1  # Learning rate; default = 0.1
momentum = 0.9  # Momentum; default = 0.9
numhid1 = 50  # Dimensionality of embedding space; default = 50
numhid2 = 200  # Number of units in hidden layer; default = 200
init_wt = 0.01  # Stddev of the initialization distribution; default = 0.01
nepochs = 20  # Number of training epochs

# Model
custom_initializer = RandomNormal(stddev=init_wt)
custom_optimizer = SGD(lr=learning_rate, momentum=momentum)

model = Sequential()
model.add(Embedding(input_dim=vocab.shape[0], output_dim=numhid1,
                    embeddings_initializer=custom_initializer,
                    input_length=input_length))
model.add(Flatten())
model.add(Dense(numhid2, kernel_initializer=custom_initializer))
model.add(Activation('sigmoid'))
model.add(Dense(vocab.shape[0], kernel_initializer=custom_initializer))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=custom_optimizer,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batchsize, epochs=nepochs, verbose=1,
          validation_data=(x_valid, y_valid))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
