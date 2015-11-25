package com.kmalik.kaggle.digitrecognizer

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel
import com.kmalik.kaggle.utils.Utils

object SimpleTrain extends Serializable {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext
    val sqc = new SQLContext(sc)
    
    val dataFile = Utils.readArg(args, "data", "/home/kapil/Rough/Kaggle/digitrecognizer/train.csv")
    val partitions = Utils.readArg(args, "partitions", 16)
    val trainRatio = Utils.readArg(args, "trainRatio", 0.7)
    val layerStr = Utils.readArg(args, "layers", "64")
    val iterations = Utils.readArg(args, "iterations", 100)
    val blockSize = Utils.readArg(args, "blockSize", 128)
    val seed = Utils.readArg(args, "seed", 13309)
    
    val data = sc.textFile(dataFile, partitions)
    			 .filter(!_.startsWith("label"))
                 .filter(_.split(",").length == 785)
                 .map(line => {
                   val features = line.split(",").map(_.toDouble)
                   (features(0), Vectors.dense(features.slice(1, 785)))
                 })            
                 
    val df = sqc.createDataFrame(data)
       			.toDF("label", "features")
    val hiddenLayers = layerStr.split(",").map(_.toInt)
    val layers = Array[Int](784) ++ hiddenLayers ++ Array[Int](10)
                 
    val splits = df.randomSplit(Array(trainRatio, 1-trainRatio), seed = seed)
    val train = splits(0).persist(StorageLevel.MEMORY_ONLY_SER)
    val test = splits(1).persist(StorageLevel.MEMORY_ONLY_SER)
        
    val trainer = new MultilayerPerceptronClassifier()
    					.setLayers(layers)
    					.setBlockSize(blockSize)
    					.setSeed(seed)
    					.setMaxIter(iterations)
    
    val model = trainer.fit(train)
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
    						.setMetricName("precision")
    
    println("Precision:" + evaluator.evaluate(predictionAndLabels))
  }
  
}