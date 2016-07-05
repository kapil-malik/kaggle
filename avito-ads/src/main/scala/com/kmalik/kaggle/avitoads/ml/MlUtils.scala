package com.kmalik.kaggle.avitoads.ml

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.WrappedArray

object MlUtils {
  
  val vectorize = udf((cols:WrappedArray[Double]) => Vectors.dense(cols.toArray))
  
  def strIndexColumns(original:DataFrame, colNames:Seq[String]) = {
    
    val indexers = colNames.map(col => (new StringIndexer()).setInputCol(col).setOutputCol(col+"Indexed").fit(original))
    
    val resultDf = indexers.foldLeft(original)((df,indexer)=> indexer.transform(df))
    
    (resultDf, indexers, colNames.map(_+"Indexed"))
  }
  
  def standardize(original:DataFrame, labelColName:String, ftColNames:Seq[String]) = {
    val labelCol = original(labelColName).as("label")
    val featuresCol = vectorize(array(ftColNames.map(x => original(x)):_*)).as("features")
    original.select(labelCol, featuresCol)
  }
  
  def standardize(original:DataFrame, ftColNames:Seq[String]) = {
    val featuresCol = vectorize(array(ftColNames.map(x => original(x)):_*)).as("features")
    original.select(featuresCol)
  }
  
}