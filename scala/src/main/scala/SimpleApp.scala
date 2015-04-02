/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object SimpleApp {
  def main(args: Array[String]) {
    //val logFile = "temp.txt" // Should be some file on your system

    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)

    sql(sc)

    //bayes(sc)
  }


  //
  // this is an example of how you might use sql to analyze a dataset
  //
  def sql(sc: SparkContext)
  {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    // A JSON dataset is pointed to by path.
    // The path can be either a single text file or a directory storing text files.
    val path = "/tmp/data/yelp_academic_dataset_review.json"

    // Create a DataFrame from the file(s) pointed to by path
    val reviews = sqlContext.jsonFile(path) //this ia a dataframe type

    // The inferred schema can be visualized using the printSchema() method.
    //reviews.printSchema()

    // Register this DataFrame as a table.
    reviews.registerTempTable("reviews")

    // SQL statements can be run by using the sql methods provided by sqlContext.
    val coolPlaces = sqlContext.sql("SELECT business_id FROM reviews WHERE votes.cool > 10")

    println(coolPlaces.count());
  }


  def bayes(sc: SparkContext)
  {
    /*
    val reviews = sc.load("../data/yelp_academic_dataset_review.json", "json")

    val reviews = sqlContext.jsonFile(path) //this ia a dataframe type

    //now we want to build a vector of features

    //we have to quantify all fields

    //here i will choose number of cool likes and I will choose 

    reviews.map( line => print(line) );

    */
  }

  def findcorpus(sc: SparkContext)
  {
/*
    val reviews = sc.load("../data/yelp_academic_dataset_review.json", "json")

    val reviews = sqlContext.jsonFile(path) //this ia a dataframe type

    // Register this DataFrame as a table.
    reviews.registerTempTable("reviews")

    // SQL statements can be run by using the sql methods provided by sqlContext.
    val coolPlaces = sqlContext.sql("SELECT business_id FROM reviews WHERE votes.cool > 10")

    println(coolPlaces.count());
    */
  }
}
