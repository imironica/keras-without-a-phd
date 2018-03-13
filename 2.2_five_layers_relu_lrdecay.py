from util import readDatabase, AccuracyHistory, showPerformance, showConfusionMatrix
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (relu)         W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (relu)         W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (relu)         W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (relu)         W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]


import tensorflow as tf
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)
np.random.seed(0)

# Read the training / testing dataset and labels
xTrain, yTrain, xTest, yTest, yLabels = readDatabase()

# Network parameters
layer1Size = 200
layer2Size = 100
layer3Size = 60
layer4Size = 30

# Train hyper-parameters
learningRate = 0.003
decay = 0.00035
noOfEpochs = 9
batchSize = 32

# Program parameters

numberOfClasses = yTrain.shape[1]
featureSize = xTrain.shape[1]

history = AccuracyHistory()
verbose = 1
showPlot = True

# Network architecture
model = Sequential()
model.add(Dense(input_dim=featureSize,
                kernel_initializer="uniform",
                units=layer1Size,
                activation='relu'))

model.add(Dense(units=layer2Size,
                activation="relu",
                kernel_initializer="uniform"))

model.add(Dense(units=layer3Size,
                activation="relu",
                kernel_initializer="uniform"))

model.add(Dense(units=layer4Size,
                activation="relu",
                kernel_initializer="uniform"))

model.add(Dense(numberOfClasses, kernel_initializer="uniform", activation="softmax"))

# Network training
sgd = Adam(lr=learningRate, decay=decay)
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


# Accuracy obtained:
# 0.9785