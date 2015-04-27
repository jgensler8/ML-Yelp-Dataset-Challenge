#
# CHANGE THIS PATH
#
YELPDIR=/tmp

#
# path ought to be absolute
#
CMDDIR=`pwd`
SCRIPTSDIR=$CMDDIR/scripts
LABELEDDIR=$CMDDIR/labeled-data

#
# from last time
#
#docker stop hdfsyelp
#docker rm hdfsyelp

#docker run -d \
# -P \
# --name hdfsyelp \
# besn0847/hdfs


#
# this is from the LAST time we ran the container
#
docker rm python-yelp

# --link hdfsyelp:hdfsyelp \
# -t -i \
docker run \
 -d \
 -w /tmp \
 -v $YELPDIR:/tmp/data \
 -v $SCRIPTSDIR:/tmp/scripts \
 -v $LABELEDDIR:/tmp/labeled-data \
 -p 8888:8888 \
 -e "PASSWORD=<><><><>" \
 --name python-yelp \
 ipython/scipyserver 
