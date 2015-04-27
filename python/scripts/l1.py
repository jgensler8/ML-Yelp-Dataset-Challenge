#!/usr/bin/python

from sklearn import metrics
import csv

with open('predictions.csv', 'rb') as csvfile:
    predictions = csv.reader(csvfile)
    true = []
    pred = []
    for row in predictions:
        true.append(row[2])
        pred.append(row[3])

y_true = [float(x) for x in true]
y_pred = [float(x) for x in pred]

print(metrics.classification_report(y_true, y_pred))
