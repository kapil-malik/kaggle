package com.kmalik.kaggle.avitoads

import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import com.kmalik.kaggle.avitoads.ml.MlUtils
import com.kmalik.kaggle.utils.DfUtils
import com.kmalik.kaggle.utils.Utils
import com.kmalik.kaggle.avitoads.ml.RfModeling
import org.apache.spark.ml.feature.StringIndexerModel

object SampleRfCV extends Serializable {
  
  def main(args:Array[String]):Unit = {
    val sc = new SparkContext()
    val sqc = new SQLContext(sc)
    
    val dataFile = Utils.readArg(args, "data")
    val outputDir = Utils.readArg(args, "output", dataFile+".out."+System.currentTimeMillis())
    val partitions = Utils.readArg(args, "partitions", 64)
    
    val trainRatio = Utils.readArg(args, "trainRatio", 0.8)    
    val numFolds = Utils.readArg(args, "numFolds", 5)
    
    val numTreesOpts = Utils.readArg(args, "numTreesOpts", "10")
    val maxDepthOpts = Utils.readArg(args, "maxDepthOpts", "5")
    val maxBinsOpts = Utils.readArg(args, "maxBinsOpts", "100")
    val evaluationMetric = Utils.readArg(args, "evaluationMetric", "areaUnderROC")
    val seed = Utils.readArg(args, "seed", 13309)
    
    val inputs = Array(dataFile, outputDir, partitions, trainRatio, numFolds,
      numTreesOpts, maxDepthOpts, maxBinsOpts, evaluationMetric, seed)
          
    inputs.map(_.toString)
    	    .foreach(println)

    val labelColName = "isDuplicate"
    val catFtNames = 
      Seq("categoryID_1","locationID_1","regionID_1","parentCategoryID_1",
          "categoryID_2","locationID_2","regionID_2","parentCategoryID_2")
    
    // Read from master file      
    val rawDf = readCsv(sqc, dataFile, partitions)
    
    // feature processing
    val (processedDf, catFtIndexers, indexedCatFtNames) = processTrainData(rawDf, catFtNames, labelColName) 
    
    // modeling
    val numTreesOptions = numTreesOpts.split(",").map(_.toInt)
    val maxDepthOptions = maxDepthOpts.split(",").map(_.toInt)
    val maxBinsOptions = maxBinsOpts.split(",").map(_.toInt)
    
    val (model, rfModel, modelParams, metricValue) = RfModeling.runCrossValidation(
      processedDf, outputDir, seed, trainRatio, numFolds, 
      numTreesOptions, maxDepthOptions, maxBinsOptions, evaluationMetric)
          
    sc.stop
    
    inputs.map(_.toString)
    	    .foreach(println)
    	    
    modelParams.foreach(println)
  }

  private def readCsv(sqc:SQLContext, path:String, partitions:Int) = {
    val df = DfUtils.load(sqc, path, 
        "com.databricks.spark.csv", 
        Map("inferSchema"->"true", "header"->"true"))
    if (partitions > 0) df.coalesce(partitions) else df
  }
  
  private def processTrainData(df:DataFrame, catFtNames: Seq[String], labelColName:String) = {
    // 1. Select base features from master file          
    val df1 = df.select(labelColName, catFtNames:_*)
    df1.cache
 
    // 2. Index categorical features  
    val (df2, catFtIndexers, newFtNames) = MlUtils.strIndexColumns(df1, catFtNames)
    
    // 3. Convert to columns "label", "features"
    val df3 = MlUtils.standardize(df2, labelColName, newFtNames)
    
    (df3, catFtIndexers, newFtNames)
  }

}
