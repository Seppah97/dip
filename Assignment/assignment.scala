package assignment21

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{window, column, desc, col}


import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType, DoubleType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg}

import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.SparkSession

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}




import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansSummary}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.feature.StringIndexer

import java.io.{PrintWriter, File}


//import java.lang.Thread
import sys.process._


import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection.immutable.Range

object assignment  {
  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)
                       
  
  val spark = SparkSession.builder()
                          .appName("assignment")
                          .config("spark.driver.host", "localhost")
                          .master("local")
                          .getOrCreate()
                          
  //custom schemas for the dataframes                        
  val customSchemaD2 = StructType(Array(
      StructField("a", DoubleType, true),
      StructField("b", DoubleType, true),
      StructField("LABEL", StringType, true))
      )
      
  val customSchemaD3 = StructType(Array(
      StructField("a", DoubleType, true),
      StructField("b", DoubleType, true),
      StructField("c", DoubleType, true),
      StructField("LABEL", StringType, true))
      )
      
      
  //read data files and create dataframes from them    
      
  val dataK5D2 =  spark.read.option("header",true).schema(customSchemaD2)
                       .csv("data/dataK5D2.csv")

  val dataK5D3 =  spark.read.option("header",true).schema(customSchemaD3)
                       .csv("data/dataK5D3.csv")
  
  //change values in column LABEL to numeric                     
  val indexer = new StringIndexer().setInputCol("LABEL").setOutputCol("num(LABEL)")                     
  val dataK5D3WithLabels = indexer.fit(dataK5D2).transform(dataK5D2)
  
                       
  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    
    
    val vectorAssembler = new VectorAssembler().setInputCols(Array("a","b")).setOutputCol("features")
    
    //scale data
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    
    //create new pipeline and give vectorAssembler and scaler to its parameters
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler,scaler))
    
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    
    //transformedData.show(false)
    
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val model = kmeans.fit(transformedData)  
    //model.summary.predictions.show(400,false)
    
    
    return model.clusterCenters.map(m => (m(0), m(1)))
    
    
  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    
    val vectorAssembler = new VectorAssembler().setInputCols(Array("a","b","c")).setOutputCol("features")
    
    
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    
    
    
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler,scaler))
    
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    
    //transformedData.show
    
    
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val model = kmeans.fit(transformedData)  
    //model.summary.predictions.show(false)
    
    
    return model.clusterCenters.map(m => (m(0), m(1),m(2)))
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    val vectorAssembler = new VectorAssembler().setInputCols(Array("a","b","num(LABEL)")).setOutputCol("features")
    
    
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    
    
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler,scaler))
    
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    
    //transformedData.show(false)
    
    
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val model = kmeans.fit(transformedData)  
    //model.summary.predictions.show(false)
    
    //filter the model to find the two most fatal clusters
    val filteredModel = model.clusterCenters.filter(x=> x(2)>=0.5)
    
    //filteredModel.foreach(println)
    
    return filteredModel.map(m => (m(0), m(1)))
  
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    val vectorAssembler = new VectorAssembler().setInputCols(Array("a","b")).setOutputCol("features")
    
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler,scaler))
    
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
   
    
    
    //recursive funtion to perform the elbow method
    def calculateCost(k: Int, values: Array[(Int,Double)]): Array[(Int, Double)] = {
      val kmeans = new KMeans().setK(k).setSeed(1L)
      val model = kmeans.fit(transformedData)  
      
      //calculates cost of the cluster
      val cost = model.computeCost(transformedData)
      
      val item = Array[(Int,Double)]((k,cost))
      
      val newValues = values ++ item
      
      //if this is the last mean to calculate, return the array
      if(k==high){
        //newValues.foreach(println)
        return newValues
      }
      
     
      else{
        //newValues.foreach(println)
        return calculateCost(k+1,newValues) 
    }
    }
    
    //Call for a function, which returns cost of the means
    val meansCost = calculateCost(low, Array[(Int,Double)]())
    
    //imports to plot the data to a figure
    import breeze.linalg._
    import breeze.numerics._
    import breeze.plot._
    
    val costValues = meansCost.map(m => m._2)
    
    val clusterValues = meansCost.map(m => m._1.toDouble)
    
    //plotting the data to a figure
    val fig = Figure()
    val p = fig.subplot(0)
    val cost = new DenseVector(costValues)
    val clusterAmount = new DenseVector(clusterValues)
    p += plot(clusterAmount, cost)
    p.title = "Elbow method"
    p.xlabel = "Amount of clusters"
    p.ylabel = "Cost of clusters"
    
    //press enter to finish the program after showing the figure
    scala.io.StdIn.readLine()
    
    return meansCost
    
  }
     
  
    
}


