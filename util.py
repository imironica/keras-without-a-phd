import pandas as pd
import os.path
import zipfile
import keras
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
import matplotlib.pylab as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))


def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def showConfusionMatrix(yLabels, predictedValues):
    predictedLabels = np.argmax(predictedValues, axis=1)
    matrix = confusion_matrix(y_true=yLabels, y_pred=predictedLabels)
    plotConfusionMatrix(matrix,
                        classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                        title='Confusion matrix')


def showPerformance(accuracy, loss, noOfEpochs, history, plot=False):
    print('\nTest accuracy:', accuracy)
    print('\nTest loss:', loss)
    if plot is True:
        plt.plot(range(1, noOfEpochs + 1), history.acc)
        plt.plot(range(1, noOfEpochs + 1), history.loss)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()


# Load the data
def unzipFile(fileToUnzip, folderToUnzip):
    print(fileToUnzip)
    with zipfile.ZipFile(fileToUnzip, "r") as zip_ref:
        zip_ref.extractall(folderToUnzip)


def readDatabase(reshape=False, categoricalValues=True):
    folderDb = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

    if not os.path.exists(os.path.join(folderDb, 'mnist_train.csv')):
        print("Unzip train file ...")
        zipFilenameTrain = os.path.join(os.path.join(folderDb, 'mnist_train.zip'))
        unzipFile(zipFilenameTrain, folderDb)

        print("Unzip test file ...")
        zipFilenameTest = os.path.join(os.path.join(folderDb, 'mnist_test.zip'))
        unzipFile(zipFilenameTest, folderDb)

    print('Reading dataset ...')
    dfTrain = pd.read_csv("./dataset/mnist_train.csv", header=None)
    dfTest = pd.read_csv("./dataset/mnist_test.csv", header=None)

    # ============================
    # Preprocess the training data
    yTrain = dfTrain[0]
    yTrainCategorical = to_categorical(yTrain, num_classes=10)

    # Drop 'label' column
    xTrain = dfTrain.drop(labels=[0], axis=1)
    # Scale between 0 and 1
    xTrain = xTrain / 255.0

    # ============================
    # Preprocess the test data
    yTest = dfTest[0]
    yTestCategorical = to_categorical(yTest, num_classes=10)
    # Drop 'label' column
    xTest = dfTest.drop(labels=[0], axis=1)
    # Scale between 0 and 1
    xTest = xTest / 255.0
    # free some space
    del dfTest
    del dfTrain

    if reshape and categoricalValues:
        xTrain = xTrain.values.reshape(-1, 28, 28, 1)
        xTest = xTest.values.reshape(-1, 28, 28, 1)
        return xTrain, yTrainCategorical, xTest, yTestCategorical, yTest

    if categoricalValues:
        return xTrain.as_matrix(), yTrainCategorical, xTest.as_matrix(), yTestCategorical, yTest

    return xTrain.as_matrix(), yTrain, xTest.as_matrix(), yTest, None


def displayImagesAndLabels(images, labels):
    labels = labels.tolist()
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1

    for label in unique_labels:
        # Pick the first image for each label.

        image = images[labels.index(label)]
        plt.subplot(4, 3, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1

        image = image.reshape(28, 28)
        _ = plt.imshow(image, cmap='gray')
    plt.show()

def displayLabelImages(images, labels, label):
    """Display images of a specific label."""
    labels = labels.tolist()
    limit = 48  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1
    indexes = [i for i in range(0,len(labels)) if labels[i] == label][0:limit]

    for index in indexes:
        image = images[index]
        plt.subplot(6, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        image = image.reshape(28, 28)
        plt.imshow(image)
    plt.show()

