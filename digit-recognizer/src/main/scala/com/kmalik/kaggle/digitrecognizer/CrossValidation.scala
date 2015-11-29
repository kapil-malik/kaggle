package com.kmalik.kaggle.digitrecognizer

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SQLContext
import com.kmalik.kaggle.utils.Utils
import org.apache.spark.sql.functions._

object CrossValidation {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext
    val sqc = new SQLContext(sc)
    
    val dataFile = Utils.readArg(args, "data")
    val predictFile = Utils.readArg(args, "predict", null)
    val outputDir = Utils.readArg(args, "output", dataFile+".out."+System.currentTimeMillis())
    val partitions = Utils.readArg(args, "partitions", 9)
    val trainRatio = Utils.readArg(args, "trainRatio", 0.8)
    val numFolds = Utils.readArg(args, "numFolds", 10)
    
    val hiddenLayerOpts = Utils.readArg(args, "hiddenLayerOpts", "128")
    val iterationOpts = Utils.readArg(args, "iterationOpts", "100")
    val blockSizeOpts = Utils.readArg(args, "blockSizeOpts", "512")

    val seed = Utils.readArg(args, "seed", 13309)
    
    val inputs = Array(dataFile, outputDir, partitions, trainRatio, numFolds, 
      hiddenLayerOpts, iterationOpts, blockSizeOpts, seed)
      
    inputs.map(_.toString)
    	  .foreach(println)
        
    val df = DRUtils.loadLabelledDf(sc, sqc, dataFile, partitions)
    val splits = df.randomSplit(Array(trainRatio, 1-trainRatio), seed = seed)
    val train = splits(0)
    val test = splits(1)

    val layerOptions = hiddenLayerOpts.split(",")
    	   							  .map(_.toInt)
    								  .map(x=> Array[Int](784,x,10))
    
    val iterationOptions = iterationOpts.split(",").map(_.toInt)
    val blockSizeOptions = blockSizeOpts.split(",").map(_.toInt)
    
    val ann = new MultilayerPerceptronClassifier()
    				.setSeed(seed)
    				
    val paramGrid = new ParamGridBuilder()
    					.addGrid(ann.layers, layerOptions)
    					.addGrid(ann.maxIter, iterationOptions)
    					.addGrid(ann.blockSize, blockSizeOptions)
    					.build()

    val evaluator = new MulticlassClassificationEvaluator()
    						.setMetricName("precision")
    					
    val cvSplit = new CrossValidator()
    					.setEstimator(ann)
    					.setEstimatorParamMaps(paramGrid)
    					.setEvaluator(evaluator)
    					.setNumFolds(numFolds)    					
    					    					
    val model = cvSplit.fit(train)
    val modelDetails = getDetails(model)
    sc.parallelize(modelDetails, 1)
      .saveAsTextFile(outputDir+"/ModelDetails")
    
    // Test the model on test set -
    val testResult = model.transform(test)
    val testPredictionLabels = testResult.select("prediction", "label")
    val testPrecision = evaluator.evaluate(testPredictionLabels)
    
    sc.parallelize(Array[String](s"TestPrecision : $testPrecision"), 1)
      .saveAsTextFile(outputDir+"/TestPrecision")
              
    DRUtils.saveDf(testPredictionLabels, outputDir+"/TestPredictions")
    
    if (predictFile != null) {
      val predictSet = DRUtils.loadUnlabelledDf(sc, sqc, predictFile, partitions)
      val predictResult = model.transform(predictSet)
      val predictions = predictResult.select("prediction")
                                     .withColumn("ImageId", monotonicallyIncreasingId())
                                     .withColumnRenamed("prediction", "Label")
                                     .select("ImageId", "Label")
      DRUtils.saveDf(predictions, outputDir+"/Predictions")
    }
    				   
    inputs.map(_.toString)
    	  .foreach(println)
        
    modelDetails.foreach(println)
  }

  private def getDetails(model:CrossValidatorModel):Array[String] = {
    val bestParams = model.bestModel.parent.extractParamMap.toSeq
    val bestBlockSize = strParamValue(bestParams, "blockSize")
    val bestMaxIter = strParamValue(bestParams, "maxIter")
    val bestFeaturesCol = strParamValue(bestParams, "featuresCol")
    val bestLabelCol = strParamValue(bestParams, "labelCol")
    val bestPredCol = strParamValue(bestParams, "predictionCol")
    val bestTol = strParamValue(bestParams, "tol")
    val bestSeed = strParamValue(bestParams, "seed")    
    val bestLayers = arrayParamValue(bestParams, "layers")
    
    return Array[String](
        s"Blocksize : $bestBlockSize",
        s"Iterations : $bestMaxIter",
        s"Layers : $bestLayers",
        s"Features Col : $bestFeaturesCol",
        s"Label Col : $bestLabelCol",
        s"Prediction Col : $bestPredCol",
        s"Tolerance : $bestTol",
        s"Seed : $bestSeed"
        )
  }
  
  private def strParamValue(params:Seq[ParamPair[_]], name:String) = {
    params.find(_.param.name.equals(name)).get.value.toString
  }

  private def arrayParamValue(params:Seq[ParamPair[_]], name:String) = {
    val list = params.find(_.param.name.equals(name)).get.value.asInstanceOf[Array[_]]
    list.mkString(",")
  }
  
}