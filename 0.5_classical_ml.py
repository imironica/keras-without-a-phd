from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from util import readDatabase

# Read the training / testing dataset and labels
xTrain, yTrain, xTest, yTest, yLabels = readDatabase(reshape=False, categoricalValues=False)

computeSVM = True
computeNearestNeighbors = True
computeSGD = True
computeNaiveBayes = True
computeDecisionTrees = True
computeAdaboost = True
computeGradientBoosting = True
computeRandomForest = True
computeExtremellyRandomForest = True


# =================================================================================================#
# Nearest neighbor
# Train the model
if computeNearestNeighbors:
    noNeighbors = 3
    descriptorName = 'Nearest neighbors ({})'.format(noNeighbors)
    clfNB = KNeighborsClassifier(n_neighbors=noNeighbors)
    clfNB.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfNB.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
# =================================================================================================#


# Support vector machines
# Train the model
descriptorName = 'SVM RBF'
print(yTrain)
cValues = [0.0001, 0.001, 0.01, 0.1]
if computeSVM:
    for cValue in cValues:
        clfSVM = svm.SVC(C=cValue, class_weight=None, coef0=0.0,
                         gamma='auto', kernel='rbf',
                         max_iter=-1, epsilon=0.1, verbose=False)
        clfSVM.fit(xTrain, yTrain)

        # Compute the accuracy of the model
        valuePredicted = svm.predict(xTest)
        accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
        confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
        print('{}: {}'.format(descriptorName, accuracy))

# =================================================================================================#
# SGD
# Train the model
if computeSGD:
    descriptorName = 'SGD'
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
# =================================================================================================#

# Naive Bayes
# Train the model
if computeNaiveBayes:
    clf = GaussianNB()
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# =================================================================================================#

# Decision trees
# Train the model
if computeDecisionTrees:
    descriptorName = 'Decision Tree Classifier '
    clf = tree.DecisionTreeClassifier()
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# =================================================================================================#

# AdaBoost Train the model
if computeAdaboost:
    descriptorName = 'Adaboost Classifier '
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# =================================================================================================#

# GradientBoostingClassifier
# Train the model

if computeGradientBoosting:
    descriptorName = 'Gradient Boosting Classifier'
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# =================================================================================================#
# RandomForestClassifier
if computeRandomForest:
    descriptorName = 'Random Forest Classifier'
    # Train the model
    clfRF = RandomForestClassifier(n_estimators=200, criterion="gini")
    clfRF.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfRF.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# ExtremellyRandomForestClassifier
if computeExtremellyRandomForest:
    descriptorName = 'Extremelly Trees Classifier'
    # Train the model
    clfRF = ExtraTreesClassifier(n_estimators=200, criterion="gini")
    clfRF.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfRF.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
