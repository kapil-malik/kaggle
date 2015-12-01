package com.kmalik.kaggle.sfcrimes

import org.apache.spark.SparkContext
import com.kmalik.kaggle.utils.Utils
import java.util.regex.Pattern
import java.util.regex.Matcher
import java.util.Date
import java.text.SimpleDateFormat
import org.apache.spark.storage.StorageLevel

object TestCleanup extends Serializable {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext()
    
    val dataFile = Utils.readArg(args, "data", "/home/centos/kaggle/sf-crimes/test.csv")
    val outFile = Utils.readArg(args, "output", dataFile+".output."+System.currentTimeMillis())
    val partitions = Utils.readArg(args, "partitions", 9)
    
    val data = sc.textFile(dataFile, partitions)
    			 .persist(StorageLevel.MEMORY_ONLY_SER)
    
    val header = data.first
    val dateIndex = DataProcessing.index(header, DataProcessing.DATES)
    val dowIndex = DataProcessing.index(header, DataProcessing.DAY_OF_WEEK)
    
    val data2 = data.map(x => DataProcessing.addDateColumns(x, dateIndex))
    				.map(x => DataProcessing.addWeekdayColumns(x, dowIndex))
    
    data2.coalesce(1)
    	 .saveAsTextFile(outFile)
    
    sc.stop()
  }

}