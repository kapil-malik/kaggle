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

object SampleRf extends Serializable {
  
  def main(args:Array[String]):Unit = {
    val sc = new SparkContext()
    val sqc = new SQLContext(sc)
    
    val dataFile = Utils.readArg(args, "data")
    val outputDir = Utils.readArg(args, "output", dataFile+".out."+System.currentTimeMillis())
    val trainRatio = Utils.readArg(args, "trainRatio", 0.8)
    
    val numTrees = Utils.readArg(args, "numTrees", 10)
    val maxDepth = Utils.readArg(args, "maxDepth", 5)
    val maxBins = Utils.readArg(args, "maxBins", 100)
    val evaluationMetric = Utils.readArg(args, "evaluationMetric", "areaUnderROC")
    val seed = Utils.readArg(args, "seed", 13309)
    
    val inputs = Array(dataFile, outputDir, trainRatio,
      numTrees, maxDepth, maxBins, evaluationMetric, seed)
          
    inputs.map(_.toString)
    	    .foreach(println)

    val labelColName = "isDuplicate"
    val catFtNames = 
      Seq("categoryID_1","locationID_1","regionID_1","parentCategoryID_1",
          "categoryID_2","locationID_2","regionID_2","parentCategoryID_2")
    
    // Read from master file      
    val rawDf = readCsv(sqc, dataFile)
    
    // feature processing
    val (processedDf, catFtIndexers, indexedCatFtNames) = processTrainData(rawDf, catFtNames, labelColName) 
    
    // modeling
    val (model, rfModel, modelParams, metricValue) = RfModeling.runTvSplit(processedDf, null, seed, 
      trainRatio, numTrees, maxDepth, maxBins, evaluationMetric)
          
    sc.stop
    
    inputs.map(_.toString)
    	    .foreach(println)
    	    
    modelParams.foreach(println)
  }

  private def readCsv(sqc:SQLContext, path:String) = {
    DfUtils.load(sqc, path, "com.databricks.spark.csv", Map("inferSchema"->"true", "header"->"true"))  
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
