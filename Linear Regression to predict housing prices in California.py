#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:58:09 2020

@author: jonasschroeder
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import dataset from GitHub
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
housing_data = pd.read_csv(url)

# Feature engineering
housing_data["rooms_per_household"] = housing_data["total_rooms"]/housing_data["households"]
housing_data["bedroom_per_rooms"] = housing_data["total_bedrooms"]/housing_data["total_rooms"]
housing_data["population_per_household"] = housing_data["population"]/housing_data["households"]

# Data preparation and transformation using pipelines: Standardized numeric data, one-hot encoded categorical data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), 
        ('std_scaler', StandardScaler()),
])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_attribs = list(housing_data.drop("ocean_proximity", axis=1))
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs), 
        ("cat", OneHotEncoder(), cat_attribs),
])
            
# not working properly, features names missing        
housing_prepared = full_pipeline.fit_transform(housing_data)
housing_prepared_df = pd.DataFrame(housing_prepared)

housing_data.columns
column_names = num_attribs + list(set(list(housing_data["ocean_proximity"]))) # WRONG ORDER OF CATEGORY NAME!!!

housing_prepared_df.columns = column_names



# old data
housing_data.hist(bins=50, figsize=(20,15)) # plots a historgram of all numeric attributes
plt.show() 

# new transformed data
housing_prepared.hist(bins=50, figsize=(20,15)) # plots a historgram of all numeric attributes
plt.show() 



#--------------------------------------------- old code ------------------------------------------------------------------------

# Notes:
# we see that median_house_value is capped at 500k -> might exclude from training set 
# median income roughly translates in 1.0 = 10k USD
# features have different scales -> feature scaling becomes necessary
# heavy tail distributions for most features -> normalize data

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)

# create income categories: pd.cut to convert numeric to categoric
housing_data["income_cat"] = pd.cut(housing_data["median_income"], 
       bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
       labels=[1, 2, 3, 4, 5])
housing_data["income_cat"].hist()

# stratified sampling: the idea is that the sample has the same distribution of income categories as the larger dataset
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 

for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
    strat_train_set = housing_data.loc[train_index] 
    strat_test_set = housing_data.loc[test_index]

# calculate the difference in distribution
(strat_train_set["income_cat"].value_counts()/len(strat_train_set)) - (housing_data["income_cat"].value_counts()/len(housing_data))

# remove income_cat from test data
strat_train_set.drop(columns=["income_cat"], inplace=True)
strat_test_set.drop(columns=["income_cat"], inplace=True)

# plot heatmap of housing prices
# price very much related to location (close to coast)
housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
             s=housing_data["population"]/100, label="population", figsize=(10,7), 
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,) 
plt.legend()

# Pearson's r / Standard correlation of the features
corr_matrix = housing_data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes=["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing_data[attributes],figsize=(12,8))

# TODO: remove capped data

# feature engineering
housing_data["rooms_per_household"] = housing_data["total_rooms"]/housing_data["households"]
housing_data["bedroom_per_rooms"] = housing_data["total_bedrooms"]/housing_data["total_rooms"]
housing_data["population_per_household"] = housing_data["population"]/housing_data["households"]

corr_matrix = housing_data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#------------------------------------------------------------------------------------------------------------
# Preparing data for ML
#------------------------------------------------------------------------------------------------------------

# fresh copy of training set
housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy()

housing.info() # total_bedrooms has missing values

# in general, you have three strategies: remove category, fill with 0 or fill with median
# this process is called imputing
# housing.dropna(subset=["total_bedrooms"]) # option 1 
# housing.drop("total_bedrooms", axis=1) # option 2 
# median = housing["total_bedrooms"].median() # option 3 
# housing["total_bedrooms"].fillna(median, inplace=True)

# sklearn offers a handy class for that
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1) # median not working for categorical data
imputer.fit(housing_num)
imputer.statistics_

X = imputer.transform(housing_num)

# convert category "ocean_proximity" to numbers using sklearn's Ordinal Encorder
# alternatively: pandas series factorize()
# housing["ocean_proximity"].factorize()
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

housing_cat = housing[["ocean_proximity"]]
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10] # shows new category numbers
ordinal_encoder.categories_ # shows original category names

# Note: this approach is not appropriate for this case since cat_1 and cat_4 are more similar and the ranking is off
# more suitable for bad-good-best labels. We could order the categories first or we use dummy variables
# with one-hot encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder =   OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray() # one-hot encoding matrix
cat_encoder.categories_ # category names

# Define your own transformer class
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room 
    def fit(self, X, y=None):
        return self # nothing else to do 
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix] 
        population_per_household = X[:, population_ix] / X[:, households_ix] 
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# Feature Scaling
# ML usually doesnt work well with features that have very differing ranges
# methods often used: standardization adn min-max scaling (normalization)

# Min-Max scaling
from sklearn.preprocessing import minmax_scale

housing_medianincome_normalized = minmax_scale(housing["median_income"])
test = pd.concat([housing.reset_index()["median_income"], pd.Series(housing_medianincome_normalized)], axis=1)

# Standardization: for each instance, remove mean value and divide by standard deviation
# advantage: less sensitive to outliers
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

housing_medianincome_standardized = scaler.fit_transform(housing["median_income"].values.reshape(-1, 1))
#test = pd.concat([housing.reset_index()["median_income"], pd.Series(housing_medianincome_standardized)], axis=1)

# Pipelines: define a pipeline consisting of various imputers, transformers etc and end with an estimator
# pipelines are especially important for proper hyperparameter tuning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), 
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
])
    
housing_num_tr = num_pipeline.fit_transform(housing_num)

# sklearn's ColumnTransformer can be used on dataframes incl numerical and categorical values
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs), 
        ("cat", OneHotEncoder(), cat_attribs),
])
    
housing_prepared = full_pipeline.fit_transform(housing)


#------------------------------------------------------------------------------------------------------------
# Train the models on prepared data
#------------------------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# assess model validity
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)

diff = (housing_predictions-housing_labels)/housing_labels

lin_mse = mean_squared_error(housing_labels, housing_predictions) 
lin_rmse = np.sqrt(lin_mse)
lin_rmse #68628 -> typical prediction error of almost $67k!

# using k-fold cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10) 
linreg_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
    
display_scores(linreg_rmse_scores)

# you should save every model incl the hyperparameters and the trained parameters
# this can be done using Python's pickle module and sklearn.externals.joblib
'''
from sklearn.externals import joblib
import joblib    
joblib.dump(lin_reg, "linreg_test.pkl")
lin_reg_loaded = joblib.load("linreg_test.pkl")    
'''

# Assess the learning curve for different sample sizes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) 
    train_errors, val_errors = [], []
    
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m]) 
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val) 
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict)) 
        val_errors.append(mean_squared_error(y_val, y_val_predict))
        
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train") 
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")    
    
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, housing_prepared, housing_labels)   
    
    
    