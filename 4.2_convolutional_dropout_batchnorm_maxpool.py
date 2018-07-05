from util import readDatabase, AccuracyHistory, showPerformance, showConfusionMatrix
from keras.models import Sequential
from keras.layers.core import Dropout, Flatten, Activation
from keras.layers import Dense, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
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
xTrain, yTrain, xTest, yTest, yLabels = readDatabase(reshape=True)

# Network parameters
firstConvLayerDepth = 4
secondConvLayerDepth = 8
thirdConvLayerDepth = 12
numberOfNeuronsFlatLayer = 512
dropoutFlatLayer = 0.2

kernelSize = (3, 3)
# Training hyperparameters
learningRate = 0.001
noOfEpochs = 20
batchSize = 32

numberOfClasses = yTrain.shape[1]
featureSize = xTrain.shape[1]

# Program parameters

history = AccuracyHistory()
verbose = 1
showPlot = False

# Network architecture

model = Sequential()

model.add(Conv2D(32, kernelSize, input_shape=(28, 28, 1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

model.add(Conv2D(32, kernelSize))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernelSize))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))

model.add(Conv2D(64, kernelSize))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(numberOfNeuronsFlatLayer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropoutFlatLayer))
model.add(Dense(numberOfClasses))

model.add(Activation('softmax'))

sgd = Adam(lr=learningRate)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=xTrain,
          y=yTrain,
          epochs=noOfEpochs,
          batch_size=batchSize,
          verbose=verbose,
          callbacks=[history])

(loss, accuracy) = model.evaluate(xTest, yTest)

showPerformance(accuracy, loss, noOfEpochs, history, plot=showPlot)

if showPlot:
    predictedValues = model.predict(xTest, batch_size=1)
    showConfusionMatrix(yLabels, predictedValues)

K.clear_session()
# Acuracy: 99.48%
