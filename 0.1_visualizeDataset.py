import matplotlib.pyplot as plt
from util import displayImagesAndLabels, readDatabase, displayLabelImages
import seaborn as sns
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--verbose", required=False, help="show images")
args = vars(ap.parse_args())

verbose = args["verbose"]

if verbose is None:
    verbose = False
else:
    verbose = bool(verbose)

sns.set(style='white', context='notebook', palette='deep')

xTrain, yTrain, xTest, yTest, yLabels = readDatabase(reshape=True, categoricalValues=False)

barValues = yTrain.value_counts()

print("\nNumber of training dataset: ")
print(xTrain.shape)
print("\nNumber of of images per label: ")
print(barValues)
print("\nNumber of test dataset: ")
print(xTest.shape)

if verbose:
    g = sns.countplot(yTrain)
    plt.show()

    g = sns.countplot(yTest)
    plt.show()

    displayImagesAndLabels(xTrain, yTrain)

    for i in range(0,10):
        displayLabelImages(xTrain, yTrain, i)





