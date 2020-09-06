#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:51:07 2020

@author: jonasschroeder
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import LinearSVC

# Load dataset
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica

# Plot data
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
plt.title('Petal length x width')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()




#------------------------------------------------------------------------------
# Maximum Margin Classifier
#------------------------------------------------------------------------------

# fill

#------------------------------------------------------------------------------
# Soft Margin Classifier / Support Vector Classifier
#------------------------------------------------------------------------------

# Define the model (here: soft margin classifier with a c=1, 
# very large margin but probably more margin violations)
svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")), 
        ])

# Fit model
svm_clf.fit(X, y)

# Use model to predict
svm_clf.predict([[5.5, 1.7]]) # prediction: array 1, meaning versicolor

# plot
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_clf.decision_function(xy).reshape(XX.shape)
# plot decision boundary and margins -> not calculated!!!
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors -> doesnt work
ax.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

# alternative visualization (no margin  nor support vectors either)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X, 
                      y=y.astype(np.integer),
                      clf=svm_clf, 
                      legend=2)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('SVM Decision Region Boundary', size=16)


#------------------------------------------------------------------------------
# Support Vector Machine
#------------------------------------------------------------------------------
