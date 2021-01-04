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
mnist = fetch_openml("mnist_784", version=1)

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
# various performance scores: precision, recall, f1, ROC
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

# Precision and Recall Scores
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


# SGDClassifier uses a decision threshold upon which the scores depends
# plot precision and recall scores as a function of the threshold

from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# PR curves are prefered when positive cases are rare like in this case of 5s
# or in cases where you care more about the false positives than the false negaives
# Good models have a curve close to the rop right corner
# interpretation: here's room for improvement
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precisions")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recalls")
    plt.legend()
    plt.ylabel("Precision/Recall Score")
    plt.xlabel("Decision Threshold Value")
    plt.title("Precision-Recall-Tradeoff")
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# Plot Precision score against recall score
plt.plot(recalls, precisions)
plt.xlabel("Recall Score")
plt.ylabel("Precision Score")
plt.title("Precision-Recall-Tradeoff")

# Find threshold mathematically
# e.g. 90% precision
threshold_90_prediction = thresholds[np.argmax(precisions >= 0.90)] #  ~1518

y_train_pred_90 = (y_scores >= threshold_90_prediction)

precision_score(y_train_5, y_train_pred_90) # 0.90012
recall_score(y_train_5, y_train_pred_90) # 0.6666

# Thus we can create classifiers for virtually any precision score, however at a cost of recall
# "If someone says, let's reach 99% precision, you should ask, at what recall."

# ROC Curve - receiver operating characteristic
# plots true-positive rate (i.e, recall) against false-positive rate

from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], "k--") # dashed diagonal
    plt.ylabel("True Positive Rate (Recall)")
    plt.xlabel("False Positive Rate")
    plt.title("ROC Curve")

plot_roc_curve(fpr, tpr)
plt.show()


#---------------------------------------------------------------------------------------------
# Binary classifier for Digit 5
# Stochastic Gradient Descent vs Random Forest
# Calculate AUC and plot ROC curves
#---------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score

mnist = fetch_openml("mnist_784", version=1, as_frame=True)

X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Stochastic Gradient Descent
sgd_clf = SGDClassifier()

cv_sgd = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy", verbose=True)
#print(cv_sgd)
print("mean cv score for SGD: " + str(cv_sgd.mean())) #0.96106

sgd_clf.fit(X_train, y_train_5)

y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

precisions_sgd, recalls_sgd, thresholds_sgd = precision_recall_curve(y_train_5, y_scores_sgd)

precision_score_sgd = round(precision_score(y_test_5, sgd_clf.predict(X_test)),4)
recall_score_sgd = round(recall_score(y_test_5, sgd_clf.predict(X_test)),4)
print("SGD has a PRECISION of " + str(precision_score_sgd) + " and a RECALL of " +  str(recall_score_sgd))


fpr_sgd, tpr_sgd, threshold_sgd = roc_curve(y_train_5, y_scores_sgd)

auc_sgd = roc_auc_score(y_train_5, y_scores_sgd)
print("AUC for SGD is: " + str(auc_sgd) + " Perfect would be 1, totally random 0.5") #0.9585

# Random Forest Classifier
rnd_clf = RandomForestClassifier(random_state=42)

cv_forest = cross_val_score(rnd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#print(cv_forest)
print("mean cv score for Random Forest: " + str(cv_forest.mean())) #0.9870

rnd_clf.fit(X_train, y_train_5)

y_probas_forest = cross_val_predict(rnd_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]

precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

precision_score_forest = round(precision_score(y_test_5, rnd_clf.predict(X_test)),4)
recall_score_forest = round(recall_score(y_test_5, rnd_clf.predict(X_test)),4)
print("Random Forest has a PRECISION of " + str(precision_score_forest) + " and a RECALL of " +  str(recall_score_forest))

fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

auc_forest = roc_auc_score(y_train_5, y_scores_forest)
print("AUC for Random Forest is: " + str(auc_forest) + " Perfect would be 1, totally random 0.5") #0.99834

# Plot ROC curve for both classifiers
# Interpretation: Random Forest classifier much better than SGD since ROC curve closer to top left corner
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], "k--") # dashed diagonal
    plt.ylabel("True Positive Rate (Recall)")
    plt.xlabel("False Positive Rate")
    plt.title("ROC Curve")
    
plt.plot(fpr_sgd, tpr_sgd, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

# PR curve for both classifiers
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, ax):
    sns.lineplot(thresholds, precisions[:-1], marker=".", color="b", label="Precisions", ax=ax)
    sns.lineplot(thresholds, recalls[:-1], marker="o", color="g", label="Recalls", ax=ax)
    plt.legend()
    plt.ylabel("Precision/Recall Score")
    plt.xlabel("Decision Threshold Value")
    plt.title("Precision-Recall Curve")

fig, axs = plt.subplots(1, 2, figsize=(10,8))
plot_precision_recall_vs_threshold(precisions_sgd, recalls_sgd, thresholds_sgd, axs.flatten()[0])
plot_precision_recall_vs_threshold(precisions_forest, recalls_forest, thresholds_forest, axs.flatten()[1])
plt.show()

#---------------------------------------------------------------------------------------------
# Multiclass Classifier
# Native multiclass classifiers are Logistic Regression, Random Forest, Naive Bayes
# Strict binary classifiers are SGD classifier and SVM
# However, there are strategies to use these for multiclass predictions (OvR, OvO)
# and scikit-learn implementations automatically run these, depending on the algorithm
#---------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score

mnist = fetch_openml("mnist_784", version=1, as_frame=True)

X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:3000], X[3000:5000], y[:3000], y[3000:5000]


# One-versus-Rest
# Train 10 binary classifiers, one per digit, and select class with the highest score
# Prefered for most binary classifiers


# One-versus-One
# Train one classifier per pair (e.g., (0,1), (0,2), ...)
# This results in way more classifiers to be trained in OvO than OvR but with less training data per classifier


# Support Vector Machine Classifier (SVC) automatically runs OvR by default since 0.19
from sklearn.svm import SVC

svm_clf = SVC(verbose=True, cache_size=1000)
svm_clf.fit(X_train, y_train) # label: 10 digits / very very slow

some_digit = X_test.iloc[0,]
some_digit_act = y_test[3000]
some_digit_pred = svm_clf.predict([some_digit])
print("Model predicted digit to be " + str(some_digit_pred) + " and it is actually a " + str(some_digit_act))





