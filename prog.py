# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:38:17 2019

@author: abhijeet
"""

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('Final_dataset_backup.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder0 = LabelEncoder()
X[:, 0] = labelencoder0.fit_transform(X[:, 0])
labelencoder1 = LabelEncoder()
X[:, 1] = labelencoder1.fit_transform(X[:, 1])
labelencoder2 = LabelEncoder()
X[:, 2] = labelencoder2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap
X = X[:, 1:]

'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
'''
'''
# Backward Elimination with adj-r squared
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = x[:, :]
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, range(len(X[0]))]
X = backwardElimination(X_opt, SL)
'''

# K fold cross validation
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20)
scores = cross_val_score(regressor, X, y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20)
regressor.fit(X, y)

'''
# Predicting a new result
y_pred = regressor.predict(X_test)

# Calculating co-relation
from sklearn.metrics import r2_score
r2_score = r2_score(y_test, y_pred)
'''

# Saving model
from sklearn.externals import joblib
joblib.dump(regressor, 'model.sav')
