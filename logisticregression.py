# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:04:23 2019

@author: nohaw
"""

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#Compute confusion matrix to evaluate the accuracy of a classification.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report
import itertools

churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()

#Lets select some features for the modeling.
#Also we change the target data type to be integer, 
#as it is a requirement by the skitlearn algorithm:
#.astype('int') -> convert type to int
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

#number of row and col
churn_df.shape

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

y = np.asarray(churn_df['churn'])
y [0:5]

#we normalize the dataset:
#z = (x - u) / s where u is mean and s is standard deviation 
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


#we split our dataset into train and test set:

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

    
#This function implements logistic regression and can use different 
#numerical optimizers to find parameters
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

#we can predict using our test set:

yhat = LR.predict(X_test)
yhat

#predict_proba returns estimates for all classes, 
#ordered by the label of classes. So, the first column is the probability
#of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):
#probability of 2 classes on same item

yhat_prob = LR.predict_proba(X_test)
yhat_prob

#Lets try jaccard index for accuracy evaluation
#size of the intersection divided by the size of the union of two label sets.
#If the entire set of predicted labels for a sample strictly match with 
#the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

jaccard_similarity_score(y_test, yhat)

#Another way of looking at accuracy of classifier is to look at confusion matrix.


def plot_confusion_matrix(cm, classes,
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
    print(confusion_matrix(y_test, yhat, labels=[1,0]))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


#15 = 1, it predict 6 correct and 9 incorrect
#25 = 0, it predict 24 correct and 1 incoorect














