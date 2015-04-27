#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, cross_validation

#select file
labeledCSV = "/tmp/labeled-data/feature_averages_final_normalized.csv"

data = pd.DataFrame.from_csv(labeledCSV)
X = data[['service_svm','location_svm','price_svm']].values
Y = data['rating'].values

FITINT = True

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( X, Y, test_size=0.2, random_state=0)

#
# Regression
#

#
## Linear
#
linreg = linear_model.LinearRegression(fit_intercept=FITINT)
linreg.fit(X_train, Y_train)
accuracy = linreg.score(X_test, Y_test)
results = linreg.predict(X_test)
for x in sorted(zip(results,Y_test), key = lambda t: t[0]):
    print(x)
    
print("LinRef", linreg.coef_, "Intercept", linreg.intercept_, "Accuracy", round(accuracy,3))
 
#
## Ridge
#
for alpha in [round(x * 0.1,1) for x in range(0, 10)]:
    ridge = linear_model.Ridge(alpha = alpha, fit_intercept=FITINT)
    ridge.fit(X_train,Y_train)
    accuracy = ridge.score(X_test, Y_test)
    print("Alpha:", alpha, "Ridge", ridge.coef_, "Accuracy", round(accuracy,3))
#
## Lasso
#
for alpha in [round(x*.0001,4) for x in range(1, 500,10)]:
    lasso = linear_model.Lasso(alpha = alpha, max_iter = 10000, fit_intercept=FITINT, tol= 1e20)
    lasso.fit(X_train,Y_train)
    accuracy = lasso.score(X_test, Y_test)
    print("Alpha:", alpha, "Lasso", lasso.coef_, "Intercept", lasso.intercept_, "Accuracy", round(accuracy,3))
    