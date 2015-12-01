package com.kmalik.kaggle.sfcrimes

import java.text.SimpleDateFormat
import java.util.regex.Pattern
import java.util.regex.Matcher

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
  
  def index(header:String, column:String):Int = header.split(",",-1).indexOf(column)
    
  def addDateColumns(line:String, dateIndex:Int):String = {
    val dateStr = line.split(",", -1)(dateIndex)
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
  
  def addWeekdayColumns(line:String, dayOfWeekIndex:Int):String = {
    val dayOfWeek = line.split(",",-1)(dayOfWeekIndex)
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