from util import readDatabase, AccuracyHistory, showPerformance, showConfusionMatrix
from keras.layers.core import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import argparse
import tensorflow as tf
from keras import backend as K

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)
np.random.seed(0)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--verbose", required=False, help="show images (0 = False, 1 = True)")
args = vars(ap.parse_args())

verbose = args["verbose"]

if verbose is None:
    verbose = False
else:
    if verbose == '1':
        verbose = True
    else:
        verbose = False

print("Verbose".format(verbose))

# Read the training / testing dataset and labels
xTrain, yTrain, xTest, yTest, yLabels = readDatabase()

# Network parameters
learningRate = 0.003

noOfEpochs = 10
batchSize = 32

numberOfClasses = yTrain.shape[1]
featureSize = xTrain.shape[1]

history = AccuracyHistory()

showPlot = verbose

# Network architecture
model = Sequential()
model.add(Dense(input_dim=featureSize,
                kernel_initializer="uniform",
                units=numberOfClasses))
model.add(Activation('softmax'))

# Network training
sgd = SGD(lr=learningRate)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=xTrain,
          y=yTrain,
          epochs=noOfEpochs,
          batch_size=batchSize,
          verbose=1,
          callbacks=[history])

(loss, accuracy) = model.evaluate(xTest, yTest)

showPerformance(accuracy, loss, noOfEpochs, history, plot=showPlot)

if showPlot:
    predictedValues = model.predict(xTest, batch_size=1)
    showConfusionMatrix(yLabels, predictedValues)

K.clear_session()
# Accuracy obtained:
# 0.9187
