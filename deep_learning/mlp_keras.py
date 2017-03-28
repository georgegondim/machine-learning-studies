import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 2000

# input image dimensions
img_rows, img_cols = 28, 28
img_pixels = img_rows * img_cols

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# preprocess data
x_train = x_train.reshape(x_train.shape[0], img_pixels)
x_test = x_test.reshape(x_test.shape[0], img_pixels)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# One hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# hold out validation data
idx = int((5 / 6) * x_train.shape[0])
x_val = x_train[idx:, :, :, :]
y_val = y_train[idx:, :]
x_train = x_train[:idx, :, :, :]
y_train = y_train[:idx, :]


# Model
model = Sequential()
model.add(Dense(units=1000, input_shape=(img_pixels,),
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0,
                                                                   stddev=0.3,
                                                                   seed=None)))
model.add(Activation('sigmoid'))
model.add(Dense(units=10, kernel_initializer=keras.initializers.RandomNormal(
    mean=0.0, stddev=0.3, seed=None)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
