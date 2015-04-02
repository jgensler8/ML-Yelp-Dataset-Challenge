# ML-Yelp-Dataset-Challenge
CS 491: Machine Learning

## Description

### Solution Overview
1. Pick our feature set (factors). We aim to pick around 3 to 5 features with only 2 labels (good or bad) for each feature.
2. Labeling the data. We will manually label a fraction of the data ourselves. We will use a semi supervised learning to label non-manually labeled data that the classifier predicts with high confidence.
3. Learn the importance of each feature (weights).
4. Repeat steps 2 and 3 with different “Categories” of Businesses, finding out neat difference between restaurants that might overlap in the Categories.

See below for a more detailed description.

## Dependencies

* Docker (~Docker version 1.5.0, build a8a31ef)
* sbt (~sbt launcher version 0.13.7)

## Getting Started

### Build the container

**this may take around 60 minutes**

We are creating our own container if we choose to add implementation that isn't suitable to fit in the run.sh command.

1. Ensure you are in the `docker` directory.

```
$ cd docker
```

1.5 (Optional) If permissions aren't set, make it so.

```
$ chmod +x build.sh
```

2. Build the container.

```
$ ./build
```

### Build the application

1. Ensure you are in the `scala` directory.

```
$ cd ../scala
```
OR
```
$ cd scala
```

2. Compile the project. This will download any dependencies in the project.

```
$ sbt package 
```

### Launch the container

1. Modify `run.sh` to update the directory your yelp data. Here, I have simply downloaded the dataset and left it in my Downloads folder.

```
#
# CHANGE THIS PATH
#
YELPDIR=/home/$USER/Downloads/yelp_dataset_challenge_academic_dataset/
```

2. Run the container

```
$ ./run.sh
```

This will give you a bash shell in the container.

### Running the application

```
:/tmp# /usr/local/spark/bin/spark-submit --class "SimpleApp" /tmp/scala-2.10/simple-project_2.10-1.0.jar
```

We also have the liberty to recompile our Scala program and not have to restart the container.

## Detailed Solution

1. LDA -> Topics -> best 3-5 many.
2. Estimated feature set: Food (Good/Bad), Price or Value (Underpriced(good)/Overpriced(bad)), Service (Good/Bad), Deals (like Groupon) (Good/Bad)
3. Bag of Words from labeled data -> Sentiment Analysis -> Predict the sentiment of a factor(say Food) for a particular restaurant.
4. Build a classifier with all these factors as features and predict the rating of a restaurant.
