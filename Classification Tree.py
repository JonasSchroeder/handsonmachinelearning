#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 08:36:16 2020

@author: jonasschroeder
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

iris = load_iris()

# Explore data set first?
iris.target_names
iris.target
iris.feature_names
iris.data

# Plot the features
x = iris.data[:, 2] # petal length
y = iris.data[:, 3] # petal width
plt.scatter(x, y, alpha=0.5)
plt.title('Petal length x width')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

# Prepare and fit the model
X = iris.data[:, 2:] # petal length and width 
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2) 
tree_clf.fit(X, y)

# Use model to predict probability and class of a flower with petal length of 5cm and petal width of 1.5cm
tree_clf.predict_proba([[5, 1.5]]) 
# output: array([[0. , 0.90740741, 0.09259259]])

tree_clf.predict([[5, 1.5]]) 
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
# output array([1])



