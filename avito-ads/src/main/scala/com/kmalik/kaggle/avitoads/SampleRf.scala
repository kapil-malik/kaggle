package com.kmalik.kaggle.avitoads

import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import com.kmalik.kaggle.avitoads.ml.MlUtils
import com.kmalik.kaggle.utils.DfUtils
import com.kmalik.kaggle.utils.Utils
import com.kmalik.kaggle.avitoads.ml.RfModeling

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

    // TrainMasterData header -
    // itemID_1,itemID_2,isDuplicate,generationMethod,
    // i_itemID_1,categoryID_1,title_1,description_1,images_array_1,attrsJSON_1,
    // price_1,locationID_1,metroID_1,lat_1,lon_1,
    // i_itemID_2,categoryID_2,title_2,description_2,images_array_2,attrsJSON_2,
    // price_2,locationID_2,metroID_2,lat_2,lon_2,
    // regionID_1,regionID_2,parentCategoryID_1,parentCategoryID_2
    
    val labelColName = "isDuplicate"
    val catFtNames = 
      Seq("categoryID_1","locationID_1","regionID_1","parentCategoryID_1",
          "categoryID_2","locationID_2","regionID_2","parentCategoryID_2")
          
    val trainRawDf = DfUtils.load(sqc, dataFile, "com.databricks.spark.csv", 
        Map("inferSchema"->"true", "header"->"true"))

    // Select base features from master file          
    val train1Df = trainRawDf.select(labelColName, catFtNames:_*)
    train1Df.cache
 
    // index categorical features  
    val (train2Df, newFtNames) = MlUtils.strIndexColumns(train1Df, catFtNames)
    
    // convert to columns "label", "features"
    val train3Df = MlUtils.standardize(train2Df, labelColName, newFtNames)
    
    val (model, rfModel, modelParams) = RfModeling.runTvSplit(train3Df, outputDir, seed, 
      trainRatio, numTrees, maxDepth, maxBins, evaluationMetric)
    
    sc.stop
    
    inputs.map(_.toString)
    	    .foreach(println)
    	    
    modelParams.foreach(println)
  }
  

}
