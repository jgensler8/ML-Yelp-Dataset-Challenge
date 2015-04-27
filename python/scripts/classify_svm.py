#!/usr/bin/python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from sklearn import cluster
import numpy as np

#select file
#labeledCSV = "/tmp/labeled-data/svm_averages.csv"
labeledCSV = "/tmp/labeled-data/feature_averages_final.csv"
#labeledCSV = "/tmp/labeled-data/reviews_with_features.csv"

d = pd.DataFrame.from_csv(labeledCSV)
d['service_price_svm'] = d['service_svm'] + d['price_svm']
d['service_location_svm'] = d['service_svm'] + d['location_svm']
#d['price_location_svm'] = d['price_svm']**2 + d['location_svm']

#TODO: select only top places with highest count
data = d

#x = data[['avg_location_svm','avg_service_svm','avg_price_svm']].values
x = data[['service_svm','location_svm','price_svm']].values
#x = data[['service_svm','service_price_svm','service_location_svm']].values
#x = data[['service_price_svm','service_location_svm','price_location_svm']].values
#x = data[['service_label','location_label','price_label']].values
#x = data['location_svm'].values
y = data['rating'].values

#normalize
#X = [ [round(row,2) for row in col] for col in x]
#Y = [ round(row,2) for row in y ]
#for label in np.nditer(y, op_flags=['readwrite']):
#    label[...] = round(label, 1)
    
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

    
    
from mpl_toolkits.mplot3d import axes3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

labels = ['^', 'o', 'h', '*', '+' ,'x']
colors = ["blue", "red", "orange", "yellow", "green", "purple"]

for size in range(3,6):
    c = cluster.SpectralClustering(n_clusters=size)
    c.fit(X_train)
    #accuracy = c.score(X_test, Y_test)
    #print("Size", size, "labels", c.labels_, c.cluster_centers_indices_, c.cluster_centers_)
    #print("iters", size, c.labels_)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("service")
    ax.set_ylabel("location")
    ax.set_zlabel("price")
    #ax.legend()
    
    #gather our data to be potted, add in marker and color
    all_data = zip(c.labels_, X_train, Y_train)
    groups = [(labels[label],colors[label],array,label,prediction) for (label,array,prediction) in all_data]
    # add each individual point to the graph
    for (label,color,array,label_num,prediction) in groups:
        X = array[0]
        Y = array[1]
        Z = array[2]
        ax.scatter(X, Y, Z, marker=label, c=color)
        
    plt.savefig("scatter_" + str(size) + ".png")
    
    for group in range(0,size):
        linreg = linear_model.LinearRegression(fit_intercept=FITINT)
        linreg.fit([array for (label,color,array,label_num,prediction) in groups if label_num == group], [prediction for (label,color,array,label_num,prediction) in groups if label_num == group])
        print("Size", size, "group", group, "LinReg", linreg.coef_, "Intercept", linreg.intercept_)

    


