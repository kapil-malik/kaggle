package com.kmalik.kaggle.avitoads.ml

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.sql.DataFrame
import com.kmalik.kaggle.utils.DfUtils
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.CustomBinaryClassificationEvaluator

object RfModeling {
  
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
        
    val labelConverter = new IndexToString()
                          .setInputCol("prediction")
                          .setOutputCol("predictedLabel")
                          .setLabels(labelIndexer.labels)
  
    val pipeline = new Pipeline()
                        .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
  
    val model = pipeline.fit(train)
    
    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]    
    val modelParams = persistModel(sc, rfModel, outputDir)
    
    // Test the model on test set -
    val testResult = model.transform(test)
    val evaluator = new CustomBinaryClassificationEvaluator()
						        .setLabelCol("labelIndexed")
                    .setRawPredictionCol("prediction")
                    .setMetricName(evaluationMetric)

    val testMetric = evaluator.evaluate(testResult)    
    sc.parallelize(Array[String](s"$evaluationMetric : $testMetric"), 1)
      .saveAsTextFile(outputDir+"/TestMetric_" + evaluationMetric)
              
    val testPredictionLabels = testResult.select("predictedLabel", "label")
    DfUtils.saveAsCsv(testPredictionLabels, outputDir+"/TestPredictions")
    
    (model, rfModel, modelParams)
  }
  
  private def persistModel(sc:SparkContext,
    rfModel:RandomForestClassificationModel, 
    outputDir:String): Array[String] = {
    
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