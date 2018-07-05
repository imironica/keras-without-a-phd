from util import readDatabase, AccuracyHistory, showPerformance, showConfusionMatrix
from keras.models import Sequential
from keras.layers.core import Dropout, Flatten
from keras.layers import Dense, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras import backend as K

# Neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)
# @ @ @ @ @ @ @ @ @ @   -- conv. layer
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2
#     ∶∶∶∶∶∶∶∶∶∶∶
#      \x/x\x\x/        -- fully connected layer (relu)
#       · · · ·
#       \x/x\x/         -- fully connected layer (softmax)
#        · · ·

# Read the training / testing dataset and labels
xTrain, yTrain, xTest, yTest, yLabels = readDatabase(reshape=True)

import argparse

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

# Network parameters
firstConvLayerDepth = 4
secondConvLayerDepth = 8
thirdConvLayerDepth = 12
numberOfNeurons = 200
dropoutPerLayer = 0.25

# Training hyperparameters
learningRate = 0.001
noOfEpochs = 20
batchSize = 32

numberOfClasses = yTrain.shape[1]
featureSize = xTrain.shape[1]

# Program parameters

history = AccuracyHistory()

showPlot = verbose

# Network architecture

model = Sequential()
model.add(Conv2D(firstConvLayerDepth, kernel_size=(5, 5),
                 activation='relu',
                 strides=(1, 1),
                 padding='same',
                 input_shape=(28, 28, 1)))
# output is 28x28

model.add(BatchNormalization())
model.add(Dropout(dropoutPerLayer))
model.add(Conv2D(secondConvLayerDepth, kernel_size=(5, 5),
                 activation='relu',
                 strides=(2, 2),
                 padding='same'))
# output is 14x14

model.add(BatchNormalization())
model.add(Dropout(dropoutPerLayer))
model.add(Conv2D(thirdConvLayerDepth, kernel_size=(5, 5),
                 activation='relu',
                 strides=(2, 2),
                 padding='same'))
# output is 7x7
model.add(BatchNormalization())
model.add(Dropout(dropoutPerLayer))
model.add(Flatten())
# output is 7x1

model.add(Dense(numberOfNeurons, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropoutPerLayer))
model.add(Dense(numberOfClasses, activation='softmax'))

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
# Acuracy 0.9918