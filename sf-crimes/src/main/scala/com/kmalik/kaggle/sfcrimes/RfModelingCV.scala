package com.kmalik.kaggle.sfcrimes

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.SQLContext
import com.kmalik.kaggle.utils.DfUtils
import com.kmalik.kaggle.utils.Utils
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.mllib.tree.model.RandomForestModel

object RfModelingCV extends Serializable {

  def main(args: Array[String]): Unit = {    
    val sc = new SparkContext()
    val sqc = new SQLContext(sc)
    
    val dataFile = Utils.readArg(args, "data")
    val predictFile = Utils.readArg(args, "predict", null)
    val outputDir = Utils.readArg(args, "output", dataFile+".out."+System.currentTimeMillis())
    val partitions = Utils.readArg(args, "partitions", 9)
    val trainRatio = Utils.readArg(args, "trainRatio", 0.8)
    val numFolds = Utils.readArg(args, "numFolds", 10)
    
    val numTreesOpts = Utils.readArg(args, "numTreesOpts", "10")
    val maxDepthOpts = Utils.readArg(args, "maxDepthOpts", "5")
    val maxBinsOpts = Utils.readArg(args, "maxBinsOpts", "100")
    
    val evaluationMetric = Utils.readArg(args, "evaluationMetric", "precision")    
    val seed = Utils.readArg(args, "seed", 13309)
    
    val inputs = Array(dataFile, outputDir, partitions, trainRatio, numFolds, 
      numTreesOpts, maxDepthOpts, maxBinsOpts, evaluationMetric, seed)
          
    inputs.map(_.toString)
    	    .foreach(println)
    	    
    val data = sc.textFile(dataFile, partitions)
    
    val df = DataProcessing.process(sqc, data)
    
    val splits = df.randomSplit(Array(trainRatio, 1-trainRatio), seed = seed)
    val train = splits(0)
    val test = splits(1)
                              
    val labelIndexer = new StringIndexer()
                            .setInputCol("label")
                            .setOutputCol("indexedLabel")
                            .fit(df)

    val featureIndexer = new VectorIndexer()
                              .setInputCol("features")
                              .setOutputCol("indexedFeatures")
                              .setMaxCategories(100)
                              .fit(df)

    val numTreesOptions = numTreesOpts.split(",").map(_.toInt)
    val maxDepthOptions = maxDepthOpts.split(",").map(_.toInt)
    val maxBinsOptions = maxBinsOpts.split(",").map(_.toInt)
    
    val rf = new RandomForestClassifier()
                  .setLabelCol("indexedLabel")
                  .setFeaturesCol("indexedFeatures")
    
    val paramGrid = new ParamGridBuilder()
                        .addGrid(rf.numTrees, numTreesOptions)
                        .addGrid(rf.maxDepth, maxDepthOptions)
                        .addGrid(rf.maxBins, maxBinsOptions)
                        .build()    
    
    val evaluator = new MulticlassClassificationEvaluator()
    						        .setLabelCol("indexedLabel")
                        .setPredictionCol("prediction")
                        .setMetricName(evaluationMetric)
                        
    val crossValidator = new CrossValidator()
                    					.setEstimator(rf)
                    					.setEstimatorParamMaps(paramGrid)
                    					.setEvaluator(evaluator)
                    					.setNumFolds(numFolds)
    
    val labelConverter = new IndexToString()
                          .setInputCol("prediction")
                          .setOutputCol("predictedLabel")
                          .setLabels(labelIndexer.labels)
  
    val pipeline = new Pipeline()
                        .setStages(Array(labelIndexer, featureIndexer, crossValidator, labelConverter))
  
    val model = pipeline.fit(train)
    
    val cvModel = model.stages(2).asInstanceOf[CrossValidatorModel]
    val rfModel = cvModel.bestModel.asInstanceOf[RandomForestClassificationModel]
    val modelParams = persistModel(rfModel, sc, outputDir)
    
    // Test the model on test set -
    val testResult = model.transform(test)
    val testMetric = evaluator.evaluate(testResult)    
    sc.parallelize(Array[String](s"$evaluationMetric : $testMetric"), 1)
      .saveAsTextFile(outputDir+"/TestMetric_" + evaluationMetric)
              
    val testPredictionLabels = testResult.select("predictedLabel", "label")
    DfUtils.saveAsCsv(testPredictionLabels, outputDir+"/TestPredictions")
    						        
    sc.stop()
    				   
    inputs.map(_.toString)
    	    .foreach(println)
    	    
    modelParams.foreach(println)
  }
 
  private def persistModel(rfModel:RandomForestClassificationModel, 
    sc:SparkContext, outputDir:String): Array[String] = {
    
    val modelString = rfModel.toDebugString
    val modelParams = Array[String]("RfTransformerParams",
                                    rfModel.explainParams(),
                                    "RfEstimatorParams",
                                    rfModel.parent.explainParams())
    
    sc.parallelize(Array[String](modelString), 1)
      .saveAsTextFile(outputDir+"/RfModel")
      
    sc.parallelize(modelParams, 1)
      .saveAsTextFile(outputDir+"/RfModelParams")
      
    modelParams
  }
  
}