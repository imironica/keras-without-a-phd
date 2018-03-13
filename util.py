import pandas as pd
import os.path
import zipfile
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

# Load the data
def unzipFile(fileToUnzip, folderToUnzip):
    print(fileToUnzip)
    with zipfile.ZipFile(fileToUnzip, "r") as zip_ref:
        zip_ref.extractall(folderToUnzip)


def readDatabase(reshape = False):
    folderDb = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'dataset')

    if not os.path.exists(os.path.join(folderDb,'mnist_train.csv')):
        print("Unzip train file ...")
        zipFilenameTrain = os.path.join(os.path.join(folderDb,'mnist_train.zip'))
        unzipFile(zipFilenameTrain, folderDb)

        print("Unzip test file ...")
        zipFilenameTest = os.path.join(os.path.join(folderDb, 'mnist_test.zip'))
        unzipFile(zipFilenameTest, folderDb)


    dfTrain = pd.read_csv("./dataset/mnist_train.csv", header=None)
    dfTest = pd.read_csv("./dataset/mnist_test.csv", header=None)

    # ============================
    # Preprocess the training data
    yTrain = dfTrain[0]
    yTrain = to_categorical(yTrain, num_classes=10)

    # Drop 'label' column
    xTrain = dfTrain.drop(labels=[0], axis=1)
    # Scale between 0 and 1
    xTrain = xTrain / 255.0

    # ============================
    # Preprocess the test data
    yTest = dfTest[0]
    yTest = to_categorical(yTest, num_classes=10)
    # Drop 'label' column
    xTest = dfTest.drop(labels=[0], axis=1)
    # Scale between 0 and 1
    xTest = xTest / 255.0
    # free some space
    del dfTest
    del dfTrain

    if reshape:
        xTrain = xTrain.values.reshape(-1, 28, 28, 1)
        xTest = xTest.values.reshape(-1, 28, 28, 1)
        return xTrain, yTrain, xTest, yTest

    return xTrain.as_matrix(), yTrain, xTest.as_matrix(), yTest