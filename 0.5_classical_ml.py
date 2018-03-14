from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from util import readDatabase

# Read the training / testing dataset and labels
xTrain, yTrain, xTest, yTest, yLabels = readDatabase(reshape=False, categoricalValues=False)

computeNearestNeighbors = True
computeSVM = True
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
    print("Train {}".format(descriptorName))
    clfNB = KNeighborsClassifier(n_neighbors=noNeighbors)
    clfNB.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfNB.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
# =================================================================================================#

# =================================================================================================#
# SGD
# Train the model
if computeSGD:
    descriptorName = 'SGD'
    print("Train {}".format(descriptorName))
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
# =================================================================================================#

# Naive Bayes
if computeNaiveBayes:
    descriptorName = 'Naive Bayes'
    print("Train {}".format(descriptorName))
    clf = GaussianNB()
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# =================================================================================================#

# Decision trees
if computeDecisionTrees:
    descriptorName = 'Decision Tree Classifier '
    print("Train {}".format(descriptorName))
    clf = tree.DecisionTreeClassifier()
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# =================================================================================================#

# AdaBoost model
if computeAdaboost:
    descriptorName = 'Adaboost Classifier '
    print("Train {}".format(descriptorName))
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# =================================================================================================#

# Gradient Boosting Classifier
if computeGradientBoosting:
    descriptorName = 'Gradient Boosting Classifier'
    print("Train {}".format(descriptorName))
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# =================================================================================================#

# Random Forest Classifier
if computeRandomForest:
    descriptorName = 'Random Forest Classifier'
    print("Train {}".format(descriptorName))
    # Train the model
    clfRF = RandomForestClassifier(n_estimators=200, criterion="gini")
    clfRF.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfRF.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))

# Extremelly RandomForest Classifier
if computeExtremellyRandomForest:
    descriptorName = 'Extremelly Trees Classifier'
    print("Train {}".format(descriptorName))
    # Train the model
    clfRF = ExtraTreesClassifier(n_estimators=200, criterion="gini")
    clfRF.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfRF.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))


# Support vector machines
descriptorName = 'SVM Linear'
cValues = [0.01, 0.1, 1, 10]
if computeSVM:
    for cValue in cValues:
        descriptorName = 'Linear SVM with C={} '.format(cValue)
        print("Train {}".format(descriptorName))
        clfSVM = svm.SVC(C=cValue, kernel='linear', verbose=False)
        clfSVM.fit(xTrain, yTrain)

        # Compute the accuracy of the model
        valuePredicted = clfSVM.predict(xTest)
        accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
        confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
        print('{}: {}'.format(descriptorName, accuracy))


descriptorName = 'SVM RBF'
cValues = [0.01, 0.1, 1, 10]
if computeSVM:
    for cValue in cValues:
        descriptorName = 'SVM with C={} '.format(cValue)
        print("Train {}".format(descriptorName))
        clfSVM = svm.SVC(C=cValue, class_weight=None,
                         gamma='auto', kernel='rbf',
                         verbose=False)
        clfSVM.fit(xTrain, yTrain)

        # Compute the accuracy of the model
        valuePredicted = clfSVM.predict(xTest)
        accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
        confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
        print('{}: {}'.format(descriptorName, accuracy))


'''
# Obtained results
    Nearest neighbors (3): 0.9705
    SGD: 0.8985
    Train Naive Bayes
    Naive Bayes: 0.5558
    Train Decision Tree Classifier
    Decision Tree Classifier : 0.879
    Train Adaboost Classifier
    Adaboost Classifier : 0.7296
    Train Gradient Boosting Classifier
    Gradient Boosting Classifier: 0.6615
    Train Random Forest Classifier
    Random Forest Classifier: 0.9704
    Train Extremelly Trees Classifier
    Extremelly Trees Classifier: 0.9735
    
    Linear SVM with C=0.01 : 0.9443
    Linear SVM with C=0.1 : 0.9472
    Linear SVM with C=1 : 0.9404
    Linear SVM with C=10 : 0.931
    
    RBF SVM with C=0.01 : 0.835
    RBF SVM with C=0.1 : 0.9166
    RBF SVM with C=1 : 0.9446
    RBF SVM with C=10 : 0.9614
'''