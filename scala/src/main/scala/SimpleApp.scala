/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.mllib.clustering.{LDA}

//import org.apache.spark.mllib.feature.{HashingTF}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF,Tokenizer}
import org.apache.spark.mllib.linalg.{SparseVector,DenseVector,Vector}

import org.apache.log4j.Logger
import org.apache.log4j.Level

import com.databricks.spark.csv._

import scala.reflect.runtime.{universe=>ru}

import org.apache.spark.sql.Row
import scala.beans.BeanInfo

@BeanInfo
case class LabeledDocument(id: Long, text: String, label: Double, b_id: String)

@BeanInfo
case class Document(id: Long, text: String)

object SimpleApp {

  val businessPath = "/tmp/data/yelp_academic_dataset_business.json"
  val reviewPath = "/tmp/data/yelp_academic_dataset_review.json"
  val businessParquetFile = "yelp_business.parquet"
  val reviewParquetFile = "yelp_review.parquet"
  val userPath = "/tmp/data/yelp_academic_dataset_user.json"
  val userParquetFile = "yelp_user.parquet"
  val serviceCsvFile = "/tmp/service_db_noquote.csv"
  val predictionCsvFile = "/tmp/predictions.csv"
  val businessRatingsCsvFile = "/tmp/business_ratings.csv"
  val businessWeightedService = "/tmp/business_service_weight_and_count.csv"

  // This is category to isolate for stage2B
  val category = "Mexican"

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.WARN);
    Logger.getLogger("akka").setLevel(Level.OFF);
    Logger.getLogger("parquet").setLevel(Level.OFF);

    case class Config (
      stage1: Boolean = false,
      stage2A: Boolean = false,
      stage2B: Boolean = false,
      stage4: Boolean = false,
      stage6A: Boolean = false,
      stage6B: Boolean = false,
      stage7: Boolean = false
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
      opt[Unit]("stage4") action { (_, c) =>
        c.copy(stage4 = true) } text("Run stage4: read parquet file, filter out data, write to .csv")
      opt[Unit]("stage6A") action { (_, c) =>
        c.copy(stage6A = true) } text("Run stage6A: launch LDA pipeline on reviews")
      opt[Unit]("stage6B") action { (_, c) =>
        c.copy(stage6B = true) } text("Run stage6B: launch Logistic Regression pipeline on service.csv")
      opt[Unit]("stage7") action { (_, c) =>
        c.copy(stage7 = true) } text("Run stage7: join labeled data with restaurant csv")

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

        if(config.stage4) stage4(sc)

        if(config.stage6A) stage6A(sc)

        if(config.stage6B) stage6B(sc)

        if(config.stage7) stage7(sc)

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

    sqlContext
    .jsonFile(reviewPath)
    .saveAsParquetFile(reviewParquetFile)

    sqlContext
    .jsonFile(userPath)
    .saveAsParquetFile(userParquetFile)
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

    val businesses = sqlContext
    .parquetFile(businessParquetFile)
    .select("categories")
    .explode("categories", "category")(explode_helper)
    .groupBy("category")
    .count()
    .sort( new Column("count").desc ) //I couldn't figure out the $ syntax, might be macro
    .limit(200)
    .saveAsCsvFile("categories.csv")
    //.save("categories.csv", "com.databricks.spark.csv")
  }

  def stage3A(sc: SparkContext)
  {
    //convert CSV back to parquet
    //TODO
  }

  // 172
  // fast food, airizona, > 20 reviews
  // businessID
  // delete newline (13 in ascii)
  // carraige return (10 in ascii)
  // --> spaces
  // , (comma) --> semicolon ()

  // nmf

  // users: userid, elite

  // reviews: text, rating, business_id, user_id

  def stage4(sc: SparkContext)
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

    val users = sqlContext
    .parquetFile(userParquetFile)
    .select("user_id", "elite")

    val reviews = sqlContext
    .parquetFile(reviewParquetFile)
    .select("business_id", "user_id", "stars", "text")

    // .join(reviews)

    val businesses = sqlContext
    .parquetFile(businessParquetFile)
    .select("name", "categories", "review_count", "state", "business_id")
    .where( new Column("state").equalTo("AZ") )
    .explode("categories", "category")(explode_helper)
    .where( new Column("category").contains("Fast Food") )
    .where( new Column("review_count").gt(20) )
    .sort("name")
    .join(users)
    .join(reviews)
    .saveAsCsvFile("businesses.csv")
    //.save("categories.csv", "com.databricks.spark.csv")
  }


  def stage6A(sc: SparkContext)
  {

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    case class WordVec(
      reviewID: scala.Long,
      features: org.apache.spark.mllib.linalg.Vector)

    val explode_helper = (text: Seq[String]) =>
    {
      text
    } : TraversableOnce[String]

    val reviews = sqlContext
    .parquetFile(reviewParquetFile)
    .limit(2000)

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(10000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    val pipe1 = tokenizer
      .transform(reviews)
    val pipe2 = hashingTF
      .transform(pipe1)

    val map_helper = (row : Row ) =>
    {
      val sv = row(1).asInstanceOf[SparseVector]
      //val dv = new DenseVector(sv.toArray)
      sv
    } : (Vector)

    val documents =
    pipe2
      .select("review_id", "features")
      .map( map_helper )
      .zipWithIndex.map(_.swap)
      .cache()

    val ldaModel = new LDA()
      .setK(100)
      .run(documents)

    println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
    val topics = ldaModel.topicsMatrix
    for (topic <- Range(0, 10)) {
      print("Topic " + topic + ":")
      for (word <- Range(0, 5)) { print(" " + topics(word, topic)); }
      println()
    }
   
  }


  def stage6B(sc: SparkContext)
  {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._

    val map_helper = (row : Row ) =>
    {
      val review = row(1).toString

      val label = (row(2) match {
        case "1" => 1D
        case "0" => 0D
        case _ => 0D
      })

      LabeledDocument(0L,review,label,row(0).toString)
    } : LabeledDocument

    val servicecsv = sqlContext
      .csvFile(serviceCsvFile)
      .map(map_helper)

    val split = servicecsv.randomSplit(Array(.8,.2))

    val training = split(0)
      .toDF()

    val testing = split(1)
      .toDF()

    val trainingCount = training.count()
    val testingCount = testing.count()
    val servicecsvCount = servicecsv.count()

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(5000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(5000)
      .setRegParam(0.01)
      .setFeaturesCol(hashingTF.getOutputCol)
      .setLabelCol("label")
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))
    
    val model = pipeline.fit(training)
    val results = model.transform(testing)

    results
      .select("id","text","label","prediction","b_id")
      .saveAsCsvFile(predictionCsvFile, Map("header" -> "true"))

    results.printSchema()

    val map_checker = (row: Row) => {
      (row(2) == row(8) match {
        case true => 1
        case false => 0
      })
    } : Int

    val right_wrong = results.map(map_checker)
    val sum      = right_wrong.sum()
    val count    = right_wrong.count()
    val accuracy = sum/count;

    println(s"count = $count, accuracy = $accuracy")
    println(s"testingCount = $testingCount, trainingCount = $trainingCount, servicecsvCount = $servicecsvCount")

  }

  def stage7(sc: SparkContext)
  {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val predictions = sqlContext
      .csvFile(predictionCsvFile)

    predictions.printSchema()
    predictions.show()

    val businessRatings = sqlContext
      .csvFile(businessRatingsCsvFile)

    businessRatings.printSchema()
    businessRatings.show()

    val joined = predictions
      .join(businessRatings, (predictions("b_id") === businessRatings("business_id")), "inner")
      .groupBy("business_id")
      .agg(Map("label" -> "avg", "prediction" -> "count"))
      .repartition(1)
      .saveAsCsvFile(businessWeightedService, Map("header" -> "true"))
    
  }

}
