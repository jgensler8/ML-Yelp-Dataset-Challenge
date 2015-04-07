name := "ML-UIC"

version := "1.0"

scalaVersion := "2.10.4"

resolvers += "sonatype-releases" at "https://oss.sonatype.org/content/repositories/releases/"

resolvers += Resolver.url("artifactory", url("http://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases"))(Resolver.ivyStylePatterns)

resolvers += "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/"

resolvers += "Spark Package Main Repo" at "https://dl.bintray.com/spark-packages/maven"

// these are provided by our docker continer
libraryDependencies += "org.apache.spark" % "spark-core_2.10" % "1.3.0" % "provided"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.3.0" % "provided"

// these will be built in our "uber"/"mega"/"fat"-jar
// plus, if we were to deploy on multiple working nodes, all of the
// working nodes would need these libraries
libraryDependencies += "com.github.scopt" % "scopt_2.10" % "3.3.0"
libraryDependencies += "com.databricks" % "spark-csv_2.10" % "1.0.3"