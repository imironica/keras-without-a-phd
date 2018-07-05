from util import readDatabase, AccuracyHistory, showPerformance, showConfusionMatrix
from keras.optimizers import Adam
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

# Read the training / testing dataset and labels
xTrain, yTrain, xTest, yTest, yLabels = readDatabase()

# Network parameters
learningRate = 0.003
layer1Size = 50

noOfEpochs = 10
batchSize = 100

numberOfClasses = yTrain.shape[1]
featureSize = xTrain.shape[1]

history = AccuracyHistory()
showPlot = verbose

# Network architecture
model = Sequential()
model.add(Dense(input_dim=featureSize,
                kernel_initializer="uniform",
                units=layer1Size,
                activation='sigmoid'))

model.add(Dense(numberOfClasses,
                kernel_initializer="uniform",
                activation="softmax"))

# Network training
sgd = Adam(lr=learningRate)
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
# 0.9676
