import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

# hyper-parameters
batch_size = 128
num_classes = 10
epochs = 2000

# input image dimensions
img_rows, img_cols = 28, 28

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# verify dimensions ordering
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# preprocess data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# one hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# hold out validation data
idx = int((5 / 6) * x_train.shape[0])
x_val = x_train[idx:, :, :, :]
y_val = y_train[idx:, :]
x_train = x_train[:idx, :, :, :]
y_train = y_train[:idx, :]

print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validaion samples')
print(x_test.shape[0], 'test samples')


# model
model = Sequential()
model.add(Conv2D(filters=6,
                 kernel_size=(5, 5),
                 input_shape=input_shape,
                 data_format=K.image_data_format()))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=16,
                 kernel_size=(5, 5),
                 data_format=K.image_data_format()))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=0)

print('\nTest loss:', score[0])
print('Test accuracy:', score[1])
