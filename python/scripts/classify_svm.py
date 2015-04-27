#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, cross_validation, cluster
from mpl_toolkits.mplot3d import axes3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#select file
labeledCSV = "/tmp/labeled-data/feature_averages_final_normalized.csv"

data = pd.DataFrame.from_csv(labeledCSV)
#d['service_price_svm'] = d['service_svm'] + d['price_svm']
#d['service_location_svm'] = d['service_svm'] + d['location_svm']
#d['price_location_svm'] = d['price_svm']**2 + d['location_svm']

#X = data[['avg_location_svm','avg_service_svm','avg_price_svm']].values
X = data[['service_svm','location_svm','price_svm']].values
#X = data[['service_svm','service_price_svm','service_location_svm']].values
#X = data[['service_price_svm','service_location_svm','price_location_svm']].values
#X = data[['service_label','location_label','price_label']].values
Y = data['rating'].values

FITINT = True

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( X, Y, test_size=0.2, random_state=0)

#
# Regression
#
'''
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
'''

#
# Clustering
#
labels = ['^', 'o', 'h', '*', '+' ,'x']
colors = ["blue", "yellow", "green", "red", "black", "purple"]

for size in range(3,6):
    #fit the training data to clusters
    c = cluster.SpectralClustering(n_clusters=size)
    c.fit(X_train)
    
    #set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("service")
    ax.set_ylabel("location")
    ax.set_zlabel("price")
    
    #gather our data to be potted
    all_data = zip(c.labels_, X_train, Y_train)
    all_data_normalized = [ {"array":array,"label":label,"prediction":prediction} for (label,array,prediction) in all_data]
    
    #calculate a regression fit for each cluster
    for cluster_label in range(0,size):
        c = filter(lambda point: point["label"] == cluster_label, all_data_normalized) 
        linreg = linear_model.LinearRegression(fit_intercept=FITINT)
        X_s = [point["array"] for point in c]
        Y_s = [point["prediction"] for point in c]
        linreg.fit( X_s, Y_s)
        print("Size", size, "Cluster", cluster_label, "Coef", linreg.coef_, "Intercept", linreg.intercept_)
        X = map(lambda x: x[0], X_s)
        Y = map(lambda x: x[1], X_s)
        Z = map(lambda x: x[2], X_s)
        Label = np.average(Y_s)
        ax.scatter( X, Y, Z, marker=labels[cluster_label], c=colors[cluster_label], label=Label)
     
    #show the default legend which we have labeled above
    ax.legend()
    
    plt.savefig("scatter_" + str(size) + ".png")
    


