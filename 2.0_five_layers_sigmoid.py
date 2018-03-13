from util import readDatabase
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid)      W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid)      W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid)      W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

# Read the training / testing dataset and labels
xTrain, yTrain, xTest, yTest = readDatabase()


layer1Size = 200
layer2Size = 100
layer3Size = 60
layer4Size = 30
learningRate = 0.01

noOfEpochs = 30
batchSize = 32

numberOfClasses = yTrain.shape[1]
featureSize = xTrain.shape[1]

model = Sequential()
model.add(Dense(input_dim=(28, 28),
                kernel_initializer="uniform",
                units=layer1Size,
                activation='sigmoid'))

model.add(Dense(units=layer2Size,
                activation="sigmoid",
                kernel_initializer="uniform"))

model.add(Dense(units=layer3Size,
                activation="sigmoid",
                kernel_initializer="uniform"))

model.add(Dense(units=layer4Size,
                activation="sigmoid",
                kernel_initializer="uniform"))

model.add(Dense(numberOfClasses, kernel_initializer="uniform", activation="softmax"))

sgd = SGD(lr=learningRate)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=xTrain, y=yTrain, epochs=noOfEpochs, batch_size=batchSize, verbose=1)

(loss, accuracy) = model.evaluate(xTest, yTest)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
