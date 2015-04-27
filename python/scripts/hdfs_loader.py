#!/usr/bin/python

import pandas as pd
import tables

unlabeledJSON = "/tmp/data/yelp_academic_dataset_review.json"
#pd.io.json.read_json(unlabeledJSON).to_hdfs("hdfs://yelphdfs")

labeledCSV = "/tmp/labeled-data/predictions.csv"
pd.DataFrame.from_csv(labeledCSV).to_hdf("hdfs://yelphdfs","yelphdfs")