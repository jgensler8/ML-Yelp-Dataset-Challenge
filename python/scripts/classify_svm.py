#!/usr/bin/python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from sklearn import cluster
import numpy as np

#labeledCSV = "/tmp/labeled-data/svm_averages.csv"
labeledCSV = "/tmp/labeled-data/feature_averages_final.csv"
#labeledCSV = "/tmp/labeled-data/reviews_with_features.csv"

d = pd.DataFrame.from_csv(labeledCSV)

#TODO: select only top places with highest count
data = d

#x = data[['avg_location_svm','avg_service_svm','avg_price_svm']].values
x = data[['service_svm','location_svm','price_svm']].values
#x = data[['service_label','location_label','price_label']].values
#x = data['location_svm'].values
y = data['rating'].values

#normalize
#X = [ [round(row,2) for row in col] for col in x]
#Y = [ round(row,2) for row in y ]
for label in np.nditer(y, op_flags=['readwrite']):
    label[...] = round(label, 1)
    
#unnormalize
X = x
Y = y

FITINT = True

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( X, Y, test_size=0.2, random_state=0)

linreg = linear_model.LinearRegression(fit_intercept=FITINT)
linreg.fit(X_train, Y_train)
accuracy = linreg.score(X_test, Y_test)
results = linreg.predict(X_test)
for x in sorted(zip(results,Y_test), key = lambda t: t[0]):
    print(x)
    
print("LinRef", linreg.coef_, "Intercept", linreg.intercept_, "Accuracy", round(accuracy,3))
   
for alpha in [round(x * 0.1,1) for x in range(0, 10)]:
    ridge = linear_model.Ridge(alpha = alpha, fit_intercept=FITINT)
    ridge.fit(X_train,Y_train)
    accuracy = ridge.score(X_test, Y_test)
    print("Alpha:", alpha, "Ridge", ridge.coef_, "Accuracy", round(accuracy,3))
    
for alpha in [round(x*.0001,4) for x in range(1, 500,10)]:
    lasso = linear_model.Lasso(alpha = alpha, max_iter = 10000, fit_intercept=FITINT, tol= 1e20)
    lasso.fit(X_train,Y_train)
    accuracy = lasso.score(X_test, Y_test)
    print("Alpha:", alpha, "Lasso", lasso.coef_, "Intercept", lasso.intercept_, "Accuracy", round(accuracy,3))

    
for size in range(3,6):
    c = cluster.SpectralClustering(n_clusters=size)
    c.fit(X_train)
    #accuracy = c.score(X_test, Y_test)
    #print("Size", size, "labels", c.labels_, c.cluster_centers_indices_, c.cluster_centers_)
    print("iters", size, c.labels_)
    for x in sorted( zip(c.labels_, Y_train), key = lambda t: t[0]):
        print(x)
    

from mpl_toolkits.mplot3d import axes3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("location")
ax.set_ylabel("service")
ax.set_zlabel("price")

X = data['location_svm'].values
Y = data['service_svm'].values
Z = data['price_svm'].values

ax.scatter(X, Y, Z)

plt.savefig("scatter.png")
