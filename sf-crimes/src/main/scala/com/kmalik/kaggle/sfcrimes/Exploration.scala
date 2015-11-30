package com.kmalik.kaggle.sfcrimes

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import java.util.regex.Pattern
import java.util.regex.Matcher
import com.kmalik.kaggle.utils.Utils

object Exploration extends Serializable {
  
  val DOW = "__DOW__"
  val WEEKDAYS = Array[String]("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
  val QUOTES_PATTERN = Pattern.compile("\"(.*?)\"")
  val OR = "__OR__"
    
  def main(args:Array[String]):Unit = {
    println("Hello SF Crimes")

    val sc = new SparkContext()
    val dataFile = Utils.readArg(args, "data", "/home/centos/kaggle/sf-crimes/train.csv")
    
    val train = sc.textFile(dataFile, 9)
    
    val header = train.first
    val trainData = train.filter(!_.startsWith("Dates")).cache
    
    val train1 = trainData.map(findDow)
    
    val comma11 = train1.map(_.split(DOW)(0).split(",",-1).length)
    val comma12 = train1.map(_.split(DOW)(1).split(",",-1).length)    
    val comma11Hist = hist(comma11)
    val comma12Hist = hist(comma12)
    
    val quotes1 = train1.map(_.split(DOW)(0).split(",",-1).length)
    val quotes2 = train1.map(_.split(DOW)(1).split(",",-1).length)
    val quotes1Hist = hist(quotes1)
    val quotes2Hist = hist(quotes2)    
    
    val train2 = train1.map(qRep)
    
    val comma21 = train2.map(_.split(DOW)(0).split(",",-1).length)
    val comma22 = train2.map(_.split(DOW)(1).split(",",-1).length)
    val comma21Hist = hist(comma21)
    val comma22Hist = hist(comma22)
    
    sc.stop()
    
    println("COMMA 11")
    comma11Hist.foreach(println)
    println("COMMA 12")
    comma12Hist.foreach(println)
    println("Quotes 11")
    quotes1Hist.foreach(println)
    println("Quotes 12")
    quotes2Hist.foreach(println)
    println("COMMA 21")
    comma21Hist.foreach(println)
    println("COMMA 22")
    comma22Hist.foreach(println)
  }
 
  def repAll(str:String,rep:String,finds:Array[String]):String = { 
    finds.foldLeft(str)( (a,b)=> a.replace(b,rep) ) 
  }

  def findDow(str:String):String = repAll(str, DOW, WEEKDAYS)
  
  def hist(rdd:RDD[_]) = { 
    rdd.map(x=>(x,1))
       .reduceByKey(_+_)
       .collect
       .sortBy(-_._2) 
  }
  
  def qRep(str:String):String = {
    val m = QUOTES_PATTERN.matcher(str); 
    val sb = new StringBuffer(); 
    while(m.find()) { 
      m.appendReplacement(sb, Matcher.quoteReplacement(m.group(1).replace(",",OR))) 
    }; 
    m.appendTail(sb); 
    sb.toString()
  }
}