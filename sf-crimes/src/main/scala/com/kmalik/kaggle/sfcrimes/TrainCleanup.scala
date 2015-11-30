package com.kmalik.kaggle.sfcrimes

import org.apache.spark.SparkContext
import com.kmalik.kaggle.utils.Utils
import java.util.regex.Pattern
import java.util.regex.Matcher
import java.util.Date
import java.text.SimpleDateFormat
import org.apache.spark.storage.StorageLevel

object TrainCleanup extends Serializable {

  val QUOTES_PATTERN = Pattern.compile("\"(.*?)\"")
  val OR = "__OR__"
  val DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
  val DATES = "Dates"
  val CATEGORY = "Category"
  val DESCRIPT = "Descript"
  val DAY_OF_WEEK = "DayOfWeek"
  val PD_DISTRICT = "PdDistrict"
  val RESOLUTION = "Resolution"
  val ADDRESS = "Address"
  val X_COORD = "X"
  val Y_COORD = "Y"
  
  val HEADER = "Dates,Category,Descript,DayOfWeek,PdDistrict,Resolution,Address,X,Y"
  def index(column:String):Int = HEADER.split(",",-1).indexOf(column)  
    
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext()
    
    val dataFile = Utils.readArg(args, "data", "/home/centos/kaggle/sf-crimes/train.csv")
    val outFile = Utils.readArg(args, "output", dataFile+".output."+System.currentTimeMillis())
    val partitions = Utils.readArg(args, "partitions", 9)
    
    val data = sc.textFile(dataFile, partitions)
    			 .persist(StorageLevel.MEMORY_ONLY_SER)
    
    val data2 = data.map(cleanMultiValuedColumns)
    				.map(addDateColumns)
    				.map(addWeekdayColumns)
    
    data2.coalesce(1)
    	 .saveAsTextFile(outFile)
    
    sc.stop()
  }
  
  def addDateColumns(line:String):String = {
    val dateStr = line.split(",", -1)(index(DATES))
    if (dateStr.equals(DATES)) {
      return line + ",DayOfMonth,Month,Hr,Quarter"
    } else {
	    val date = DATE_FORMAT.parse(dateStr)
	    val dayOfMonth = date.getDate()
	    val month = date.getMonth()
	    val hr = date.getHours()
	    val quarter = hr/3
	    return Array(line, dayOfMonth, month, hr, quarter).mkString(",")
    }
  }
  
  def addWeekdayColumns(line:String):String = {
    val dayOfWeek = line.split(",",-1)(index(DAY_OF_WEEK))
    if (dayOfWeek.equals(DAY_OF_WEEK)) {
      return line + ",Weekend"
    } else {
      val weekend = dayOfWeek.equals("Saturday") || dayOfWeek.equals("Sunday")
      return line + "," + (if(weekend) 1 else 0)
    }
  }
  
  def cleanMultiValuedColumns(line:String):String = {
    val matcher = QUOTES_PATTERN.matcher(line); 
    val sb = new StringBuffer(); 
    while(matcher.find()) { 
      matcher.appendReplacement(sb, Matcher.quoteReplacement(matcher.group(1).replace(",",OR))) 
    }; 
    matcher.appendTail(sb); 
    sb.toString()
  }  
}