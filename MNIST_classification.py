#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:36:49 2021

Inspired by
Chapter 3: Classification


@author: jonasschroeder
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, as_frame=True)

# 784 features for 28x28 pixels
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

# Plot example

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()


# Train test split: already shuffled by openml
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


#---------------------------------------------------------------------------------------------
# Binary classifier for Digit 5
# based on a simple Stochastic Gradient Descent
#---------------------------------------------------------------------------------------------

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier()

cv_sgd = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(cv_sgd)
print("mean accuracy: " + str(cv_sgd.mean())) #0.96

# 0.96 accuracy looks great, however roughly 90% are not 5s
# hence, we have a skewed dataset and accuracy is not the right measure

# Confusion matrix instead
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) # returns predictions instead of validation score

confusion_matrix(y_train_5, y_train_pred)

# row: actual class (neg, pos)
# column: predicted class (neg, pos)
#
# true_neg, false_pos
# false_neg, true_pos
#
# -> calculate 
# precision = tp/(tp+fp) meaning: share of classified positives are actually positive
# recall = tp/(tp+fn) meaning: share of correctly classified as positive based on all actual positives, i.e. detection rate

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred) # 0.834
recall_score(y_train_5, y_train_pred) # 0.739

# F1-score: harmonic mean of precision and recall with more weights to low values
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred) # 0.784

# precision-recall tradeoff
# f1_scores are good for comparing models but some problems prefer recall and some other precision as an accuracy measure alone
# precision: adult videos filtered for children 
# recall: detection of shoplifters
 



