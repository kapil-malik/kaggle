package com.kmalik.kaggle.utils

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext

object DfUtils {
    
  def saveAsCsv(df:DataFrame, outPath:String, delim:String = ","):Unit = {
	  df.coalesce(1)
	    .write
    	.format("com.databricks.spark.csv")
    	.option("delimiter", delim)
    	.option("header", "true")
    	.save(outPath)
  }

  def checkpoint(df:DataFrame, sqc:SQLContext, outPath:String, format:String = "parquet"):DataFrame = {
    val dfPath = outPath + "/dfCp" + System.currentTimeMillis()
    format match {
                case "json" => {
                                  df.coalesce(1).write.json(dfPath)
                                  sqc.read.json(dfPath)
                                }
                case "parquet" => {
                                  df.coalesce(1).write.parquet(dfPath)
                                  sqc.read.parquet(dfPath)
                                }
                case _ => {
                            df.coalesce(1).write.save(dfPath)
                            sqc.read.load(dfPath)
                          }
              }
  }
  
}