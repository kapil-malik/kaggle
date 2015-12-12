package com.kmalik.kaggle.sfcrimes

import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import com.kmalik.kaggle.utils.Utils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.StringIndexer
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.StructField

object RfModeling extends Serializable {

  def main(args: Array[String]): Unit = {    
    val sc = new SparkContext()
    val sqc = new SQLContext(sc)
    
    val dataFile = Utils.readArg(args, "data")
    val predictFile = Utils.readArg(args, "predict", null)
    val outputDir = Utils.readArg(args, "output", dataFile+".out."+System.currentTimeMillis())
    val partitions = Utils.readArg(args, "partitions", 9)
    val trainRatio = Utils.readArg(args, "trainRatio", 0.8)

    val seed = Utils.readArg(args, "seed", 13309)
    
    val data = sc.textFile(dataFile, partitions)
    			 
    val header = data.first
    val ncol = header.split(",").length
    
    val labelIndex = DataProcessing.index(header, DataProcessing.CATEGORY)
    
    val labelData = data.filter(!_.equals(header))
    					.map(DataProcessing.cleanMultiValuedColumns)
    					.filter(_.split(",").length == ncol)
    					.map(row => {
    						val label = row.split(",")(labelIndex)
    						(label, row)
    					})
    
    val df = sqc.createDataFrame(labelData)
    			      .toDF("label","row")
    
    val fXtractor = new FeatureExtractor(header)
    val df2 = fXtractor.transform(df)
    
    val districtIndexer = new StringIndexer()
    							              .setInputCol("district")
    							              .setOutputCol("districtNum")
    							              .fit(df2)
    							            
    val dowIndexer = new StringIndexer()
							            .setInputCol("dayOfWeek")
							            .setOutputCol("dayOfWeekNum")
							            .fit(df2)
							            
		val fCombiner = new FeatureCombiner(FeatureExtractor.featureNamesAndTypes)
							            
    val pipeline = new Pipeline()
                        .setStages(Array(districtIndexer, dowIndexer, fCombiner))
    
    val df3 = pipeline.fit(df2).transform(df2)
    saveDf(df3, outputDir + "/df", ",")

    sc.stop
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