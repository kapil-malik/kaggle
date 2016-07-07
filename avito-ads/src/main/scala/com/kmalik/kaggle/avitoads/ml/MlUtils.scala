package com.kmalik.kaggle.avitoads.ml

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.WrappedArray

object MlUtils {
  
  val vectorize = udf((cols:WrappedArray[Double]) => Vectors.dense(cols.toArray))
  
  val vectorize2 = udf((cols:WrappedArray[Any]) => Vectors.dense(cols.toArray.map(_.toString.toDouble)))
  
  val safeDouble = udf((a:Any, nVal:Double) => if(a==null || a=="null") nVal else a.toString.toDouble)
  
  def strIndexColumns(original:DataFrame, colNames:Seq[String]) = {
    
    val indexers = colNames.map(col => (new StringIndexer()).setInputCol(col).setOutputCol(col+"Indexed").fit(original))
    
    val resultDf = indexers.foldLeft(original)((df,indexer)=> indexer.transform(df))
    
    (resultDf, indexers, colNames.map(_+"Indexed"))
  }
  
  def convertToDouble(original:DataFrame, colNames:Seq[String], nVal:Double = 0.0) = {
    colNames.foldLeft(original)((df, col) => {
      val df2 = df.withColumnRenamed(col, col+"Raw")
      val df3 = df2.withColumn(col, safeDouble(df2(col+"Raw"), lit(nVal)))
      df3
    })
  }
  
  def standardizeLabeled(original:DataFrame, labelColName:String, ftColNames:Seq[String], strict:Boolean = false) = {
    val labelCol = original(labelColName).as("label")
    val featuresCol = if (strict) {
      vectorize(array(ftColNames.map(x => original(x)):_*)).as("features")
    } else {
      vectorize2(array(ftColNames.map(x => original(x)):_*)).as("features")
    }
    original.select(labelCol, featuresCol)
  }
  
  def standardizeUnlabeled(original:DataFrame, ftColNames:Seq[String], strict:Boolean = false) = {
    val featuresCol = if (strict) {
      vectorize(array(ftColNames.map(x => original(x)):_*)).as("features")
    } else {
      vectorize2(array(ftColNames.map(x => original(x)):_*)).as("features")
    }
    original.select(featuresCol)
  }
  
}