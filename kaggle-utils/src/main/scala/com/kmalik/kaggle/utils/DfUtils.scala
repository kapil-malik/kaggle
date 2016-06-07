package com.kmalik.kaggle.utils

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SaveMode

object DfUtils {
    
  def load(sqc:SQLContext, location:String, format:String, 
      options:Map[String,String] = Map()):DataFrame = {
      sqc.read
         .format(format)
         .options(options)
         .load(location)
  }

  def save(df:DataFrame, path:String, format:String, partitions:Int = 128,
      options:Map[String,String] = Map()) = {
      val baseDf:DataFrame = if(partitions > 0) df.coalesce(partitions) else df
      baseDf.write
            .mode(SaveMode.Overwrite)
            .format(format)
            .options(options)
            .save(path)
  }
  
  def saveAsCsv(df:DataFrame, outPath:String, delim:String = ","):Unit = {
    save(df, outPath, "com.databricks.spark.csv", 1, 
        Map("delimiter"->delim, "header"->"true"))
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