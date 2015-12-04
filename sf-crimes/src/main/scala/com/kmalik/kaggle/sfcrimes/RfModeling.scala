package com.kmalik.kaggle.sfcrimes

import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.sql._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import com.kmalik.kaggle.utils.Utils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.StringIndexer
import scala.collection.mutable.WrappedArray

object RfModeling extends Serializable {

  def main(args: Array[String]): Unit = {    
    val sc = new SparkContext()
    val sqc = new SQLContext(sc)
    
    val dataFile = Utils.readArg(args, "data")
    val predictFile = Utils.readArg(args, "predict", null)
    val outputDir = Utils.readArg(args, "output", dataFile+".out."+System.currentTimeMillis())
    val partitions = Utils.readArg(args, "partitions", 9)
    val trainRatio = Utils.readArg(args, "trainRatio", 0.8)

    val seed = Utils.readArg(args, "seed", 13309)
    
    val data = sc.textFile(dataFile, partitions)
    			 
    val header = data.first
    val ncol = header.split(",").length
    
    val labelIndex = DataProcessing.index(header, DataProcessing.CATEGORY)
    
    val labelData = data.filter(!_.equals(header))
    					.map(DataProcessing.cleanMultiValuedColumns)
    					.filter(_.split(",").length == ncol)
    					.map(row => {
    						val label = row.split(",")(labelIndex)
    						(label, row)
    					})
    
    val df = sqc.createDataFrame(labelData)
    			.toDF("label","row")
    			
    val dateIndex = DataProcessing.index(header, DataProcessing.DATES)
    val dowIndex = DataProcessing.index(header, DataProcessing.DAY_OF_WEEK)
    val districtIndex = DataProcessing.index(header, DataProcessing.PD_DISTRICT)
    val xIndex = DataProcessing.index(header, DataProcessing.X_COORD)
    val yIndex = DataProcessing.index(header, DataProcessing.Y_COORD)
    
    sqc.udf.register("strColumn",
        (row:String, index:Int) => row.split(",")(index))
    
    sqc.udf.register("numColumn",
        (row:String, index:Int) => row.split(",")(index).toDouble)
    
    sqc.udf.register("dateColumn", 
        (row:String, dateIndex:Int, index:Int) => DataProcessing.buildDateColumns(row.split(",")(dateIndex))(index))
    
    sqc.udf.register("weekend", 
        (row:String, dowIndex:Int) => DataProcessing.buildWeekend(row.split(",")(dowIndex)))
    
    sqc.udf.register("features", 
        (features:WrappedArray[Double]) => Vectors.dense(features.toArray))
        
    
    val df2 = df.select(df("label"),
    				callUDF("strColumn", df("row"), lit(districtIndex)).as("district"),
    				callUDF("strColumn", df("row"), lit(dowIndex)).as("dayOfWeek"),
    				callUDF("numColumn", df("row"), lit(xIndex)).as("xCoord"),
    				callUDF("numColumn", df("row"), lit(yIndex)).as("yCoord"),
            	 	callUDF("dateColumn", df("row"), lit(dateIndex), lit(0)).as("dayOfMonth"),
            	 	callUDF("dateColumn", df("row"), lit(dateIndex), lit(1)).as("month"),
            	 	callUDF("dateColumn", df("row"), lit(dateIndex), lit(2)).as("hr"),
            	 	callUDF("dateColumn", df("row"), lit(dateIndex), lit(3)).as("quarter"),
            	 	callUDF("weekend", df("row"), lit(dowIndex)).as("weekend")
            		)
    val districtIndexer = new StringIndexer()
    							.setInputCol("district")
    							.setOutputCol("districtNum")
    val df3 = districtIndexer.fit(df2).transform(df2)
    
    val dowIndexer = new StringIndexer()
    							.setInputCol("dayOfWeek")
    							.setOutputCol("dayOfWeekNum")
    val df4 = dowIndexer.fit(df3).transform(df3)
    
    val df5 = df4.select(df4("label"), 
    					callUDF("features",
    							array(
    							df4("districtNum"),
	    				    	df4("dayOfWeekNum"),
	    				    	df4("xCoord"),
	    				    	df4("yCoord"),
	    				    	df4("dayOfMonth"),
	    				    	df4("month"),
	    				    	df4("hr"),
	    				    	df4("quarter"),
	    				    	df4("weekend"))
	    				        ).as("features"))
    saveDf(df5, outputDir + "/df", ",")
    
    sc.stop()
  }

  
  def saveDf(df:DataFrame, outPath:String, delim:String = "\t") = {
	  df.coalesce(1)
	    .write
    	.format("com.databricks.spark.csv")
    	.option("delimiter", delim)
    	.option("header", "true")
    	.save(outPath)
  }  
}