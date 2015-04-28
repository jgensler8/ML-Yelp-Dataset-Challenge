#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, cross_validation, cluster
from mpl_toolkits.mplot3d import axes3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

#select file
labeledCSV = "/tmp/labeled-data/feature_averages_final_normalized.csv"

data = pd.DataFrame.from_csv(labeledCSV)
X = data[['name','service_svm','location_svm','price_svm']].values
Y = data['rating'].values

FITINT = True

X_train_all, X_test_all, Y_train, Y_test = cross_validation.train_test_split( X, Y, test_size=0, random_state=0)
X_train_names = [x[:1][0] for x in X_train_all]
X_train = [x[1:] for x in X_train_all]
X_test_names = [x[:1][0] for x in X_test_all]
X_test = [x[1:] for x in X_test_all]

#
# Clustering
#
labels = ['^', 'o', 'h', '*', '+' ,'x', 's', 'p', 'D', '<', '>', 'v']
colors = ["blue", "yellow", "green", "red", "black", "purple", "orange", "grey", "teal", "tan"]

for size in range(2,10):
    # Fit the training data to clusters
    c = cluster.SpectralClustering(n_clusters=size)
    c.fit(X_train)
    
    # Set up plot
    fig = plt.figure(1,figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("service")
    ax.set_ylabel("location")
    ax.set_zlabel("price")
    ax.set_title("Total Restaurants: " + str(len(X_train)))
    
    # Gather our data to be potted
    all_data = zip(c.labels_, X_train, Y_train, X_train_names)
    all_data_normalized = [ {"array":array,"label":label,"prediction":prediction,"name":name} for (label,array,prediction,name) in all_data]
    
    for cluster_label in range(0,size):
        c = filter(lambda point: point["label"] == cluster_label, all_data_normalized) 
        X_s = [point["array"] for point in c]
        Y_s = [point["prediction"] for point in c]
        
        # Fit linear for each cluster
        linreg = linear_model.LinearRegression(fit_intercept=FITINT)
        linreg.fit( X_s, Y_s)
        print("Size", size, "Cluster", cluster_label, "Coef", linreg.coef_, "Intercept", linreg.intercept_)
        #print([round(y,1) for y in sorted(Y_s)])
        
        # Graph each cluster
        X = map(lambda x: x[0], X_s)
        Y = map(lambda x: x[1], X_s)
        Z = map(lambda x: x[2], X_s)
        mostCommonNames = Counter( [point["name"].lower() for point in c] ).most_common(2)
        Label = str("Avg: " + str(round(np.average(Y_s),3))) + " Size:" + str(len(c)) + " " + str(mostCommonNames)
        ax.scatter( X, Y, Z, marker=labels[cluster_label], c=colors[cluster_label], label=Label)
     
    # Put legend to the right of graph
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, .5),
              fancybox=True, shadow=True, ncol=1)
    # Extend the picture if needed
    plt.savefig("scatter_" + str(size) + ".png",bbox_extra_artists=(lgd,), bbox_inches='tight')
    


