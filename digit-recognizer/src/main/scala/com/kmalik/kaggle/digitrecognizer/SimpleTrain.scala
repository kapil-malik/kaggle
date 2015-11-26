package com.kmalik.kaggle.digitrecognizer

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel
import com.kmalik.kaggle.utils.Utils
import org.apache.spark.sql.DataFrame

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
                    
    val df = DRUtils.loadLabelledDf(sc, sqc, dataFile, partitions)    
    val layers = DRUtils.nnLayers(layerStr)
    
    runANN(df, trainRatio, layers, blockSize, seed, iterations)
  }
  
  def runANN(df:DataFrame, trainRatio:Double, layers:Array[Int], 
    blockSize:Int, seed:Long, iterations:Int) = {
    println("Train Ratio : " + trainRatio)
    
    val splits = df.randomSplit(Array(trainRatio, 1-trainRatio), seed = seed)
    val train = splits(0)
    val test = splits(1)
        
    val trainer = new MultilayerPerceptronClassifier()
    					.setLayers(layers)
    					.setBlockSize(blockSize)
    					.setSeed(seed)
    					.setMaxIter(iterations)
    
    println(trainer.explainParams)
    					
    val model = trainer.fit(train)
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
    						.setMetricName("precision")
    
    println("Precision:" + evaluator.evaluate(predictionAndLabels))
  }
  
}