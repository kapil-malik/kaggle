package com.kmalik.kaggle.avitoads.models

import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import com.kmalik.kaggle.avitoads.ml.MlUtils
import com.kmalik.kaggle.utils.DfUtils
import com.kmalik.kaggle.utils.Utils
import com.kmalik.kaggle.avitoads.ml.RfModeling
import org.apache.spark.ml.feature.StringIndexerModel

object AvitoSimpleRfModel extends Serializable {
  
  def main(args:Array[String]):Unit = {
    val sc = new SparkContext()
    val sqc = new SQLContext(sc)
    
    val dataFile = Utils.readArg(args, "data")
    val outputDir = Utils.readArg(args, "output", dataFile+".simpleRfModel."+System.currentTimeMillis())
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

    // Read from master file      
    val rawDf = readCsv(sqc, dataFile, partitions)
    
    // feature processing
    val processedDf = processTrainData(rawDf) 
    
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
  
  private def processTrainData(df:DataFrame) = {
    val labelColName = "isDuplicate"
    
    // 1. Select base features from master file          
    val selectFtNames = 
      Seq("categoryID_1","locationID_1","regionID_1","parentCategoryID_1","price_1","lat_1","lon_1",
          "categoryID_2","locationID_2","regionID_2","parentCategoryID_2","price_2","lat_2","lon_2")
          
    val df1 = df.select(labelColName, selectFtNames:_*)
    df1.cache
 
    // 2. Add more categorical features - similarity features
    val df2 = df1.withColumn("categoryIDMatch", df1("categoryID_1")===df1("categoryID_2"))
                 .withColumn("locationIDMatch", df1("locationID_1")===df1("locationID_2"))
                 .withColumn("regionIDMatch", df1("regionID_1")===df1("regionID_2"))
                 .withColumn("parentCategoryIDMatch", df1("parentCategoryID_1")===df1("parentCategoryID_2"))
                 
    val catFtNames = 
      Seq("categoryID_1","locationID_1","regionID_1","parentCategoryID_1",
          "categoryID_2","locationID_2","regionID_2","parentCategoryID_2",
          "categoryIDMatch","locationIDMatch","regionIDMatch","parentCategoryIDMatch")
          
    // 3. Index categorical features  
    val (df3, _, newCatFtNames) = MlUtils.strIndexColumns(df2, catFtNames)
    
    // 4. Add more numeric features     
    val df4 = MlUtils.convertToDouble(df3, Seq("price_1", "price_2"), 1.0)
    val df5 = MlUtils.convertToDouble(df4, Seq("lat_1", "lat_2", "lon_1", "lon_2"), 0.0)                 
    
    val df6 = df5.withColumn("price1Log", log10(df5("price_1")))
                 .withColumn("price2Log", log10(df5("price_2")))
                 .withColumn("latDiff", df5("lat_1") - df5("lat_2"))
                 .withColumn("lonDiff", df5("lon_1") - df5("lon_2"))
    
    // 5. Convert to columns "label", "features"
    val newFtNames = newCatFtNames ++ Seq(
        "price_1","lat_1","lon_1","price1Log",
        "price_2","lat_2","lon_2","price2Log",
        "latDiff","lonDiff")                 
    val df7 = MlUtils.standardizeLabeled(df6, labelColName, newFtNames)
    
    df7
  }

}
