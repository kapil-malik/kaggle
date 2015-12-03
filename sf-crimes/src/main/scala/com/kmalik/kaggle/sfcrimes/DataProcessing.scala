package com.kmalik.kaggle.sfcrimes

import java.text.SimpleDateFormat
import java.util.regex.Pattern
import java.util.regex.Matcher
import org.apache.spark.sql.catalyst.expressions.WeekOfYear

object DataProcessing extends Serializable {
  
  val QUOTES_PATTERN = Pattern.compile("\"(.*?)\"")
  val OR = "__OR__"
    
  val DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
  
  val ID = "Id"
  val DATES = "Dates"
  val CATEGORY = "Category"
  val DESCRIPT = "Descript"
  val DAY_OF_WEEK = "DayOfWeek"
  val PD_DISTRICT = "PdDistrict"
  val RESOLUTION = "Resolution"
  val ADDRESS = "Address"
  val X_COORD = "X"
  val Y_COORD = "Y"  
  
  val DATE_COLUMNS = Array[String]("DayOfMonth","Month","Hr","Quarter")
  val WEEKEND = "Weekend"
  
  def index(header:String, column:String):Int = header.split(",",-1).indexOf(column)
    
  def buildDateColumns(dateStr:String) = {
    val date = DATE_FORMAT.parse(dateStr)
    val dayOfMonth = date.getDate()
    val month = date.getMonth()
    val hr = date.getHours()
    val quarter = hr/3
    Array[Double](dayOfMonth, month, hr, quarter)
  }
  
  def buildWeekend(dayOfWeek:String) = if (dayOfWeek.equals("Saturday") || dayOfWeek.equals("Sunday")) 1.0 else 0.0
  
  def addDateColumns(line:String, dateIndex:Int):String = {
    val dateStr = line.split(",", -1)(dateIndex)
    if (dateStr.equals(DATES)) {
      return line + "," + DATE_COLUMNS.mkString(",") 
    } else {
	    return line + "," + buildDateColumns(dateStr).mkString(",")
    }
  }
  
  def addWeekdayColumns(line:String, dayOfWeekIndex:Int):String = {
    val dayOfWeek = line.split(",",-1)(dayOfWeekIndex)
    if (dayOfWeek.equals(DAY_OF_WEEK)) {
      return line + "," + WEEKEND
    } else {
      return line + "," + buildWeekend(dayOfWeek)
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