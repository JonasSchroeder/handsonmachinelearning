#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:01:59 2020

@author: jonasschroeder
"""

# Generate random x and y values
import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Scatter plot of data
import matplotlib.pyplot as plt
plt.scatter(X, y, alpha=0.5)
plt.title('Scatter Plot of Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#----------------------------------------------------------------------------------------------------------
# First, the complicated way to show the basic concept
#----------------------------------------------------------------------------------------------------------

# Calculate Normal Equation (best weights for minimal cost)
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance 
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

# Use model for prediction
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance 
y_predict = X_new_b.dot(theta_best)
y_predict

# plot the model
plt.plot(X_new, y_predict, "r-") 
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15]) 
plt.show()

#----------------------------------------------------------------------------------------------------------
# Second, simple solution using Scikit-Learn
#----------------------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
y_predict = lin_reg.predict(X)


#----------------------------------------------------------------------------------------------------------
# Linear Regression using Gradient Descent algorithm to find optimum
#----------------------------------------------------------------------------------------------------------

rom sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1) 
sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_


