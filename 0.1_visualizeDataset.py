import pandas as pd
import matplotlib.pyplot as plt
import os.path
import zipfile

import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

# Load the data
def unzipFile(fileToUnzip, folderToUnzip):
    print(fileToUnzip)
    with zipfile.ZipFile(fileToUnzip, "r") as zip_ref:
        zip_ref.extractall(folderToUnzip)

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

yTrain = dfTrain[0]
# Drop 'label' column
xTrain = dfTrain.drop(labels=[0], axis=1)

yTest = dfTest[0]
# Drop 'label' column
xTest = dfTest.drop(labels=[0], axis=1)

# free some space
del dfTest
del dfTrain


barValues = yTrain.value_counts()

print("\nNumber of training dataset: ")
print(xTrain.shape)
print("\nNumber of of images per label: ")
print(barValues)

g = sns.countplot(yTrain)
plt.show()

print("\nNumber of test dataset: ")
print(xTest.shape)

g = sns.countplot(yTest)
plt.show()







