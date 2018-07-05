from util import readDatabase, AccuracyHistory, showPerformance, showConfusionMatrix
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.optimizers import Adam
from keras import backend as K
# Neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]

# Read the training / testing dataset and labels

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

xTrain, yTrain, xTest, yTest, yLabels = readDatabase(reshape=True)

# Network parameters
firstConvLayerDepth = 6
firstKernelSize = (5, 5)
secondConvLayerDepth = 12
secondKernelSize = (5, 5)
thirdConvLayerDepth = 24
thirdKernelSize = (5, 5)

numberOfNeurons = 200

# Training hyperparameters
learningRate = 0.001
noOfEpochs = 3
batchSize = 32

numberOfClasses = yTrain.shape[1]
featureSize = xTrain.shape[1]

# Program parameters

history = AccuracyHistory()

showPlot = verbose

# Network architecture

model = Sequential()
model.add(Conv2D(firstConvLayerDepth, kernel_size=firstKernelSize,
                 activation='relu',
                 strides=(1, 1),
                 padding='same',
                 input_shape=(28, 28, 1)))
# output is 28x28

model.add(Conv2D(secondConvLayerDepth, kernel_size=secondKernelSize,
                 activation='relu',
                 strides=(2, 2),
                 padding='same'))
# output is 14x14

model.add(Conv2D(thirdConvLayerDepth, kernel_size=thirdKernelSize,
                 activation='relu',
                 strides=(2, 2),
                 padding='same'))

# output is 7x7
model.add(Flatten())
model.add(Dense(numberOfNeurons, activation='relu'))
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
# Acuracy 0.9862