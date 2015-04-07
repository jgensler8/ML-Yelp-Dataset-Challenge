#
# CHANGE THIS PATH
#
YELPDIR=/home/$USER/Downloads/yelp_dataset_challenge_academic_dataset/

#
# path ought to be absolute
#
CMDDIR=`pwd`
WORKDIR=$CMDDIR/target/

#
# this is from the LAST time we ran the container
#
docker rm spark-yelp

docker run -t -i \
 -v $WORKDIR:/tmp/ \
 -v $YELPDIR:/tmp/data \
 -p 8090:8090 \
 -p 4040:4040 \
 -w /tmp \
 --name spark-yelp \
 spark-yelp:latest \
 bash


#/usr/local/spark/bin/spark-submit --class "SimpleApp" /tmp/scala-2.10/simple-project_2.10-1.0.jar
#
#/usr/local/spark/bin/spark-submit --packages com.databricks:spark-csv_2.10:1.0.2,com.github.scopt:scopt_2.10:3.3.0 --class "SimpleApp" /tmp/scala-2.10/ML-UIC-assembly-1.0.jar --stage2A
#
# Use the above line to run the scala program. You could either
# run this instead of bash or exec bash and then rebuild the project
# over and over without having to restart the container
#

