# Keras without a PhD 

This repository contains a Keras version of 'TensorFlow and Deep Learning without a PhD' presentation from Google. 

The lab is taking few hours to be ran and understood. The exercises take you through the design and optimisation of a neural network for recognising handwritten digits, from the simplest possible solution all the way to a recognition accuracy above 99%. It covers feed-forward (with one or more layers with different activation functions) and convolutional networks, as well as techniques such as learning rate decay and dropout. We also perform a comparison with other classical machine learning techniques, such as Nearest Neighbor, Support Vector Machines, Random Forests and gradient boosted trees.

# Table of contents
1. [Project installation](#1-project-installation)
2. [Project description](#2-project-description)  
3. [Results and learnings](#3-results-and-learnings)

# 1. Project installation
[[back to the top]](#table-of-contents)

### Installation steps (tested on Ubuntu) ###

Install GIT

*sudo apt-get install git*

Get the source code
 
*mkdir keras-without-a-phd*

*cd keras-without-a-phd*

*git init*

*git clone https://github.com/imironica/keras-without-a-phd.git*

Install latest python packages from *requirements.txt* file

*pip3 install -r requirements.txt*

# 2. Project description 
[[back to the top]](#table-of-contents)

This case study shows how to create a model for **a classical OCR problem** using th **MNIST dataset**.

# 3. Results and learnings
[[back to the top]](#table-of-contents)

We have tested a broad list of ML algorithms, starting from classical approaches *(SVM, Nearest neighbors, Naive Bayes, Random forests, Boosted trees)* to deep learning architectures *(feed forward and convolutional neural networks)*.  All presented scores were computed using the accuracy metric.

## Results with classical ML approaches
<pre>- Nearest neighbors (3): 0.9705
- Stochastic Gradient Descent: 0.8985
- Naive Bayes: 0.5558
- Decision Tree Classifier : 0.879
- Adaboost Classifier : 0.7296
- Gradient Boosting Classifier: 0.6615
- Random Forest Classifier: 0.9704
- Extremelly Trees Classifier: 0.9735
- Linear SVM with C=0.01 : 0.9443
- Linear SVM with C=0.1 : 0.9472
- Linear SVM with C=1 : 0.9404
- Linear SVM with C=10 : 0.931
- RBF SVM with C=0.01 : 0.835
- RBF SVM with C=0.1 : 0.9166
- RBF SVM with C=1 : 0.9446
- RBF SVM with C=10 : 0.9614
 </pre>
## Results with Feed Forward Neural Network architecture

<pre>- One Softmax layer: 0.9187
- Two layers: Sigmoid -> Softmax: 0.9676
- Five sigmoid layers: 0.9745
- Five relu layers layers: 0.9755
- Five relu layers with learning rate decay: 0.9785
- Five relu layers with learning rate decay and dropout: 0.9795
</pre>

## Results with Convolutional Neural Network architecture

<pre>- 3 convolutional layers + flatten + softmax: 0.9862
- 3 convolutional layers + flatten + dropout + softmax: 0.99
- 3 convolutional layers + batch normalization + flatten + dropout + softmax + learning rate decay: 99.48
</pre>