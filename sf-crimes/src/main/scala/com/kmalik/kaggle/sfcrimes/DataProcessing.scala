package com.kmalik.kaggle.sfcrimes

import java.text.SimpleDateFormat
import java.util.regex.Pattern
import java.util.regex.Matcher
import org.apache.spark.sql.catalyst.expressions.WeekOfYear
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SQLContext
import com.kmalik.kaggle.utils.Utils
import com.kmalik.kaggle.utils.DfUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

object DataProcessing extends Serializable {
  
  private val QUOTES_PATTERN = Pattern.compile("\"(.*?)\"")
  private val OR = "__OR__"

  def main(args: Array[String]):Unit = {
    val sc = new SparkContext()
    val sqc = new SQLContext(sc)
    
    val dataFile = Utils.readArg(args, "data")
    val outputDir = Utils.readArg(args, "output", dataFile+".out."+System.currentTimeMillis())
    val partitions = Utils.readArg(args, "partitions", 9)
    
    val data = sc.textFile(dataFile, partitions)
    val df = process(sqc, data)
    
    DfUtils.saveAsCsv(df, outputDir + "/df")
    
    sc.stop
  }

  def process(sqc:SQLContext, data:RDD[String]):DataFrame = {
    val header = data.first
    val ncol = header.split(",").length
    
    val labelIndex = header.split(",",-1).indexOf("Category")
    
    val labelData = data.filter(!_.equals(header))
    					.map(cleanMultiValuedColumns)
    					.filter(_.split(",").length == ncol)
    					.map(row => {
    						val label = row.split(",")(labelIndex)
    						(label, row)
    					})
    
    val df = sqc.createDataFrame(labelData)
    			      .toDF("label","row")
    
    val fXtractor = new FeatureExtractor(header)
    val df2 = fXtractor.transform(df)
    
    val districtIndexer = new StringIndexer()
    							              .setInputCol("district")
    							              .setOutputCol("districtNum")
    							              .fit(df2)
    							            
    val dowIndexer = new StringIndexer()
							            .setInputCol("dayOfWeek")
							            .setOutputCol("dayOfWeekNum")
							            .fit(df2)
							            
		val fCombiner = new FeatureCombiner(FeatureExtractor.featureNamesAndTypes)
							            
    val pipeline = new Pipeline()
                        .setStages(Array(districtIndexer, dowIndexer, fCombiner))
    
    val df3 = pipeline.fit(df2).transform(df2)
    df3
  }
  
  private def cleanMultiValuedColumns(line:String):String = {
    val matcher = QUOTES_PATTERN.matcher(line); 
    val sb = new StringBuffer(); 
    while(matcher.find()) { 
      matcher.appendReplacement(sb, Matcher.quoteReplacement(matcher.group(1).replace(",",OR))) 
    }; 
    matcher.appendTail(sb); 
    sb.toString()
  }  
  
}