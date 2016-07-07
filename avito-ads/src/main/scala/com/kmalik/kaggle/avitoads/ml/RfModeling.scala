package com.kmalik.kaggle.avitoads.ml

import org.apache.commons.lang3.StringUtils
import org.apache.spark.SparkContext
import org.apache.spark.ml.CustomBinaryClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import com.kmalik.kaggle.utils.DfUtils
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.CrossValidatorModel

object RfModeling {

// Uses binary classifier
  def runCrossValidation(df:DataFrame, outputDir:String, 
    seed:Long,
    trainRatio:Double, numFolds:Int, 
    numTreesOptions:Seq[Int], 
    maxDepthOptions:Seq[Int], 
    maxBinsOptions:Seq[Int], evaluationMetric:String) = {
    val sc = df.sqlContext.sparkContext
    val splits = df.randomSplit(Array(trainRatio, 1-trainRatio), seed = seed)
    val train = splits(0)
    val test = splits(1)
    
    val labelIndexer = new StringIndexer()
                            .setInputCol("label")
                            .setOutputCol("labelIndexed")
                            .fit(df)

    val featureIndexer = new VectorIndexer()
                              .setInputCol("features")
                              .setOutputCol("featuresIndexed")
                              .setMaxCategories(100)
                              .fit(df)

    val rf = new RandomForestClassifier()
                  .setLabelCol("labelIndexed")
                  .setFeaturesCol("featuresIndexed")
                  .setSeed(seed)
        
    val paramGrid = new ParamGridBuilder()
                        .addGrid(rf.numTrees, numTreesOptions)
                        .addGrid(rf.maxDepth, maxDepthOptions)
                        .addGrid(rf.maxBins, maxBinsOptions)
                        .build()
                        
    val evaluator = new CustomBinaryClassificationEvaluator()
						            .setLabelCol("labelIndexed")
                        .setRawPredictionCol("prediction")
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
    
    val modelParams = persistModel(sc, rfModel, outputDir)
    
    // Apply model on train set -
    val trainResult = model.transform(train)
    val trainMetricValue = evaluator.evaluate(trainResult)
    
    // Apply model on test set -
    val testResult = model.transform(test)
    val testMetricValue = evaluator.evaluate(testResult)
    
    if (StringUtils.isNotBlank(outputDir)) {
      sc.parallelize(Array[String](s"$evaluationMetric : $trainMetricValue"), 1)
        .saveAsTextFile(outputDir+"/TrainMetric_" + evaluationMetric)
        
      sc.parallelize(Array[String](s"$evaluationMetric : $testMetricValue"), 1)
        .saveAsTextFile(outputDir+"/TestMetric_" + evaluationMetric)
                
      val testPredictionLabels = testResult.select("predictedLabel", "label")
      DfUtils.saveAsCsv(testPredictionLabels, outputDir+"/TestPredictions")
    }
    
    (model, rfModel, modelParams, testMetricValue)
  }
  
// Uses binary classifier
  def runTvSplit(df:DataFrame, outputDir:String, seed:Long,
    trainRatio:Double, numTrees:Int, maxDepth: Int, maxBins:Int, evaluationMetric:String) = {
    val sc = df.sqlContext.sparkContext
    val splits = df.randomSplit(Array(trainRatio, 1-trainRatio), seed = seed)
    val train = splits(0)
    val test = splits(1)
    
    val labelIndexer = new StringIndexer()
                            .setInputCol("label")
                            .setOutputCol("labelIndexed")
                            .fit(df)

    val featureIndexer = new VectorIndexer()
                              .setInputCol("features")
                              .setOutputCol("featuresIndexed")
                              .setMaxCategories(100)
                              .fit(df)

    val rf = new RandomForestClassifier()
                  .setLabelCol("labelIndexed")
                  .setFeaturesCol("featuresIndexed")
                  .setNumTrees(numTrees)
                  .setMaxDepth(maxDepth)
                  .setMaxBins(maxBins)
                  .setSeed(seed)
        
    val labelConverter = new IndexToString()
                          .setInputCol("prediction")
                          .setOutputCol("predictedLabel")
                          .setLabels(labelIndexer.labels)
  
    val pipeline = new Pipeline()
                        .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
  
    val model = pipeline.fit(train)
    
    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]    
    val modelParams = persistModel(sc, rfModel, outputDir)
    
    val evaluator = new CustomBinaryClassificationEvaluator()
						        .setLabelCol("labelIndexed")
                    .setRawPredictionCol("prediction")
                    .setMetricName(evaluationMetric)

    // Apply model on train set -
    val trainResult = model.transform(train)
    val trainMetricValue = evaluator.evaluate(trainResult)
    
    // Apply model on test set -
    val testResult = model.transform(test)
    val testMetricValue = evaluator.evaluate(testResult)
    
    if (StringUtils.isNotBlank(outputDir)) {
      sc.parallelize(Array[String](s"$evaluationMetric : $trainMetricValue"), 1)
        .saveAsTextFile(outputDir+"/TrainMetric_" + evaluationMetric)
                
      sc.parallelize(Array[String](s"$evaluationMetric : $testMetricValue"), 1)
        .saveAsTextFile(outputDir+"/TestMetric_" + evaluationMetric)
                
      val testPredictionLabels = testResult.select("predictedLabel", "label")
      DfUtils.saveAsCsv(testPredictionLabels, outputDir+"/TestPredictions")
    }
    
    (model, rfModel, modelParams, testMetricValue)
  }
  
  def predict(model:PipelineModel, df:DataFrame, outputDir:String) = {
    
    val predictions = model.transform(df).coalesce(1)
    
    predictions.registerTempTable("predictions")
    val output = df.sqlContext.sql(
    "select monotonically_increasing_id() as id,predictedLabel as probability from predictions")
    
    DfUtils.saveAsCsv(output, outputDir+"/SubmitPredictions")
  }
  
  private def persistModel(sc:SparkContext,
    rfModel:RandomForestClassificationModel, 
    outputDir:String): Array[String] = {
    
    val modelString = rfModel.toDebugString
    val modelParams = Array[String]("RfTransformerParams",
                                    rfModel.explainParams(),
                                    "RfEstimatorParams",
                                    rfModel.parent.explainParams())
    
    if (StringUtils.isNotBlank(outputDir)) {                                
      sc.parallelize(Array[String](modelString), 1)
        .saveAsTextFile(outputDir+"/RfModel")
        
      sc.parallelize(modelParams, 1)
        .saveAsTextFile(outputDir+"/RfModelParams")
    }
    
    modelParams
  }
  
}