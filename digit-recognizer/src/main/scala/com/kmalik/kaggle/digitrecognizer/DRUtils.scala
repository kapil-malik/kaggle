package com.kmalik.kaggle.digitrecognizer

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame

object DRUtils {

  def loadLabelledDf(sc:SparkContext, sqc:SQLContext, dataFile:String, partitions:Int):DataFrame = {
    val data = sc.textFile(dataFile, partitions)
    			 .filter(!_.startsWith("label"))
                 .filter(_.split(",").length == 785)
                 .map(line => {
                   val features = line.split(",").map(_.toDouble)
                   (features(0), Vectors.dense(features.slice(1, 785)))
                 })            
                 
    sqc.createDataFrame(data)
       .toDF("label", "features")
  }

  def loadUnlabelledDf(sc:SparkContext, sqc:SQLContext, dataFile:String, partitions:Int):DataFrame = {
    val data = sc.textFile(dataFile, partitions)
    			 .filter(!_.startsWith("pixel"))
                 .filter(_.split(",").length == 784)
                 .map(line => (0,Vectors.dense(line.split(",").map(_.toDouble))) )            
                 
    sqc.createDataFrame(data)
       .toDF("label", "features")
  }
  
  def nnLayers(hiddenLayerStr:String):Array[Int] = {
    val hiddenLayers = hiddenLayerStr.split(",").map(_.toInt)
    Array[Int](784) ++ hiddenLayers ++ Array[Int](10)
  }
  
  def saveDf(df:DataFrame, outPath:String, delim:String = "\t") = {
	  df.coalesce(1)
	    .write
    	.format("com.databricks.spark.csv")
    	.option("delimiter", delim)
    	.option("header", "true")
    	.save(outPath)
  }
  
}