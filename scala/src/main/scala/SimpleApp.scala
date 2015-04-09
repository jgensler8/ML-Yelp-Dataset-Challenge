/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.expressions._

import com.databricks.spark.csv._

object SimpleApp {

  val businessPath = "/tmp/data/yelp_academic_dataset_business.json"
  val businessParquetFile = "yelp_business.parquet"

  def main(args: Array[String]) {

    case class Config (
      stage1: Boolean = false,
      stage2A: Boolean = false,
      stage2B: Boolean = false
    )

    val parser = new scopt.OptionParser[Config]("scopt") {
      head("scopt", "3.x")
      note("You must explicitly choose which stages you want to run.\n")
    
      opt[Unit]("stage1") action { (_, c) =>
        c.copy(stage1 = true) } text("Run stage1: convert Yelp JSON to parquet file")
      opt[Unit]("stage2A") action { (_, c) =>
        c.copy(stage2A = true) } text("Run stage2A: generate a csv of unqiue categories")
      opt[Unit]("stage2B") action { (_, c) =>
        c.copy(stage2B = true) } text("Run stage2B: read parquet file, filter out data, write to .csv")
      help("help") text("show help text")
    }

    // parser.parse returns Option[C]
    parser.parse(args, Config()) match {
      case Some(config) =>

        val conf = new SparkConf().setAppName("Simple Application")
        val sc = new SparkContext(conf)

        if(config.stage1) stage1(sc)

        if(config.stage2A) stage2A(sc)

        if(config.stage2B) stage2B(sc)

      case None =>
        // arguments are bad, error message will have been displayed
    }

  }

  //
  // this is an example of how you might use sql to analyze a dataset
  //
  /*
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
  */

  def stage1(sc: SparkContext)
  {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    sqlContext
    .jsonFile(businessPath)
    .saveAsParquetFile(businessParquetFile)
  }

  def stage2A(sc: SparkContext)
  {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    // for some reason, this must be declared anonymously and 
    // not inside of the curried function below
    // basically, our data (Parquet) is actually a sequence
    // but we just needed to use the explode function to 
    // "pull" that data out.
    val explode_helper = (categories: Seq[String]) =>
    {
      categories
    } : TraversableOnce[String]

    sqlContext
    .parquetFile(businessParquetFile)
    .select("categories")
    .explode("categories", "category")(explode_helper)
    .groupBy("category")
    .count()
    .sort( new Column("count").desc ) //I couldn't figure out the $ syntax, might be macro
    .limit(20)
    .saveAsCsvFile("categories.csv")
    //.save("categories.csv", "com.databricks.spark.csv")
  }


  def stage2B(sc: SparkContext)
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

  def stage3A(sc: SparkContext)
  {
    //convert CSV back to parquet
    //TODO
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
