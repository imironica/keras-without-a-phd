from util import readDatabase
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils


xTrain, yTrain, xTest, yTest = readDatabase()

model = Sequential()
model.add(Dense(input_dim=784, kernel_initializer="uniform", units=10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=xTrain, y=yTrain, epochs=10, batch_size=32, verbose=1)

(loss, accuracy) = model.evaluate(xTest, yTest)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

