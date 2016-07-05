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
import org.apache.spark.ml.feature.StringIndexer

object SampleRfSubmit extends Serializable {
  
  def main(args:Array[String]):Unit = {
    val sc = new SparkContext()
    val sqc = new SQLContext(sc)
    
    val trainDataFile = Utils.readArg(args, "trainData")
    val submitDataFile = Utils.readArg(args, "submitData", null)
    val outputDir = Utils.readArg(args, "output", trainDataFile+".out."+System.currentTimeMillis())
    val trainRatio = Utils.readArg(args, "trainRatio", 0.8)
    
    val numTrees = Utils.readArg(args, "numTrees", 10)
    val maxDepth = Utils.readArg(args, "maxDepth", 5)
    val maxBins = Utils.readArg(args, "maxBins", 100)
    val evaluationMetric = Utils.readArg(args, "evaluationMetric", "areaUnderROC")
    val seed = Utils.readArg(args, "seed", 13309)
    
    val inputs = Array(trainDataFile, submitDataFile, outputDir, trainRatio,
      numTrees, maxDepth, maxBins, evaluationMetric, seed)
          
    inputs.map(_.toString)
    	    .foreach(println)

    val labelColName = "isDuplicate"
    val catFtNames = 
      Seq("categoryID_1","locationID_1","regionID_1","parentCategoryID_1",
          "categoryID_2","locationID_2","regionID_2","parentCategoryID_2")
    
    // Read from master file      
    val trainRawDf = readCsv(sqc, trainDataFile)
    val submitRawDf = readCsv(sqc, submitDataFile)
    
    // build indexers from full data
    val (catFtIndexers, indexedCatFtNames) = buildFtIndexers(trainRawDf, submitRawDf, catFtNames)
    
    // feature processing
    val trainProcessedDf = processTrainData(trainRawDf, catFtNames, labelColName, catFtIndexers, indexedCatFtNames) 
    
    // modeling
    val (model, rfModel, modelParams, metricValue) = RfModeling.runTvSplit(trainProcessedDf, outputDir, seed, 
      trainRatio, numTrees, maxDepth, maxBins, evaluationMetric)
    
    if (submitDataFile != null) {
      // Read from master file
      val submitRawDf = readCsv(sqc, submitDataFile)
      
      // apply same feature processing
      val submitProcessedDf = processSubmitData(submitRawDf, catFtNames, catFtIndexers, indexedCatFtNames)
      
      // predict using model
      RfModeling.predict(model, submitProcessedDf, outputDir)
    }
      
    sc.stop
    
    inputs.map(_.toString)
    	    .foreach(println)
    	    
    modelParams.foreach(println)
  }

  private def readCsv(sqc:SQLContext, path:String) = {
    DfUtils.load(sqc, path, "com.databricks.spark.csv", Map("inferSchema"->"true", "header"->"true"))  
  }
  
  private def buildFtIndexers(trainDf:DataFrame, submitDf:DataFrame, ftNames: Seq[String]) = {
    // 1. Select base features from master file          
    val fullDataDf = trainDf.select(ftNames.head, ftNames.tail:_*)
                            .unionAll(submitDf.select(ftNames.head, ftNames.tail:_*))
    fullDataDf.cache
 
    val indexers = ftNames.map(col => (new StringIndexer()).setInputCol(col).setOutputCol(col+"Indexed").fit(fullDataDf))
    
    (indexers, ftNames.map(_+"Indexed"))
  }
  
  private def processTrainData(df:DataFrame, catFtNames: Seq[String], labelColName:String, 
    catFtIndexers:Seq[StringIndexerModel], indexedFtNames:Seq[String]) = {
    // 1. Select base features from master file          
    val df1 = df.select(labelColName, catFtNames:_*)
    df1.cache
 
    // 2. Index categorical features  
    val df2 = catFtIndexers.foldLeft(df1)((df,indexer)=> indexer.transform(df))
    
    // 3. Convert to columns "label", "features"
    val df3 = MlUtils.standardize(df2, labelColName, indexedFtNames)
    
    df3
  }
  
  private def processSubmitData(df:DataFrame, catFtNames: Seq[String], 
    catFtIndexers:Seq[StringIndexerModel], indexedFtNames:Seq[String]) = {
    // 1. Select base features from master file          
    val df1 = df.select(catFtNames.head, catFtNames.tail:_*)
    df1.cache
 
    // 2. Index categorical features  
    val df2 = catFtIndexers.foldLeft(df1)((df,indexer)=> indexer.transform(df))
    
    // 3. Convert to columns "features"
    val df3 = MlUtils.standardize(df2, indexedFtNames)
    
    df3
  }

}
