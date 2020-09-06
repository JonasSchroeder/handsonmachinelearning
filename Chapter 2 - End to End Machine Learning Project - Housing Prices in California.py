#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:55:42 2020

From Hand on Machine Learning Book, Chapter 2

Idea: We work in an real estate company and our boss wants us to build a model to predict the median housing prices for 
various destricts in California. We use the California Census data which inclused median income and other information.

Should we use supervised or unsupervised methods to build our system? Reinforcement learning?
-> supervised since we have a target defined which is the median housing price

Classification or regression problem? Or something else?
-> regression since our target is numeric (median housing price per destrict)

Do we have all the features necessary? Which should be choose?

https://github.com/ageron/handson-ml2


@author: jonasschroeder
"""

import pandas as pd
import matplotlib.pyplot as plt

# Import dataset from GitHub
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
housing_data = pd.read_csv(url)


#------------------------------------------------------------------------------------------------------------
# Simple linear regression (first try)
#------------------------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = housing_data["median_house_value"]
X = housing_data.drop(columns=["median_house_value"])

# check for nan
y.isnull().values.any() # -> False
y.isnull().sum() # -> 0
X.isnull().sum() # -> 207 missing values for total_bedrooms

housing_temp = housing_data[housing_data["total_bedrooms"].isnull()==False]
y = housing_temp["median_house_value"]
X = housing_temp.drop(columns=["median_house_value", "ocean_proximity"]) # ignoring categorical variable 
X = X[X["total_bedrooms"].isnull()==False] # drop instances with missing info on number of bedrooms

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
y_predict = lin_reg.predict(X)

# Model validity
print("MSE of the model is: " + str(mean_squared_error(y, y_predict))) # 483M ?!?!?
print("R-squared of the model is: " + str(r2_score(y, y_predict))) # 0.6369


#------------------------------------------------------------------------------------------------------------
# Process following the book
#------------------------------------------------------------------------------------------------------------

housing_data.info() # quick overview of the data, type, missing values etc

housing_data["ocean_proximity"].value_counts() # categorical attribute with 5 values

summary_table = housing_data.describe() # method to explore the numerical values in a dataframe

housing_data.hist(bins=50, figsize=(20,15)) # plots a historgram of all numeric attributes
plt.show() 

# Notes:
# we see that median_house_value is capped at 500k -> might exclude from training set 
# median income roughly translates in 1.0 = 10k USD
# features have different scales -> feature scaling becomes necessary
# heavy tail distributions for most features -> normalize data


