import pandas as pd
import scipy.misc
import os

# Load the data
dfTrain = pd.read_csv("./dataset/mnist_train.csv", header=None)
dfTest = pd.read_csv("./dataset/mnist_test.csv", header=None)

yTrain = dfTrain[0]
# Drop 'label' column
xTrain = dfTrain.drop(labels=[0], axis=1)

yTest = dfTest[0]
# Drop 'label' column
xTest = dfTest.drop(labels=[0], axis=1)

# free some space
del dfTest
del dfTrain

# Reshape images
xTrain = xTrain.values.reshape(-1, 28, 28)
xTest = xTest.values.reshape(-1, 28, 28)

dicTrain = {}
dicTest = {}
for i in range(0, 10):
    dicTrain[i] = 1
    dicTest[i] = 1

# Generate training jpg files
index = 0

folder = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'dataset')
folder = os.path.join(folder, 'images')
folderTrain = os.path.join(folder, 'train')

for img in xTrain:
    label = yTrain[index]
    imageName = '{}_{}.jpg'.format(label, dicTrain[label])
    imagePath = os.path.join(folderTrain, imageName)
    scipy.misc.imsave(imagePath, img)
    dicTrain[label] = dicTrain[label] + 1
    index += 1
    print('Save train {}'.format(imageName))

# Generate test jpg files
index = 0

folder = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'dataset')
folder = os.path.join(folder, 'images')
folderTest = os.path.join(folder, 'test')

for img in xTest:
    label = yTest[index]
    imageName = '{}_{}.jpg'.format(label, dicTest[label])
    imagePath = os.path.join(folderTest, imageName)
    scipy.misc.imsave(imagePath, img)
    dicTest[label] = dicTest[label] + 1
    index += 1
    print('Save test {}'.format(imageName))
