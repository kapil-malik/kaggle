package com.kmalik.kaggle.avitoads

import scala.io.Source
import java.io.File
import scala.collection.mutable.{ListBuffer => uList, Map => uMap}
import scala.util.Try
import java.io.PrintWriter

object DataPrep {

  private val FLUSH_BATCH = 10000
  
  def cleanMultilineCsv(inputPath:String, outputPath:String, batch:Int = FLUSH_BATCH) = {
    println("Start")
    val reader = Source.fromFile(new File(inputPath)).getLines()
    val writer = new PrintWriter(new File(outputPath))
    
    var _count = 0
    val buffer = uList[String]()    
    while(reader.hasNext) {
      val line = reader.next()
      _count = _count + 1
      if(isNewline(line)) {
        if (!buffer.isEmpty) {
          writer.println(buffer.mkString)
          if (_count >= batch) {
            println(s"Flushing ${_count} lines")
            writer.flush()
            _count = 0
          }
          buffer.clear()
        }
      }
      buffer+=line
    }
    
    if (!buffer.isEmpty) {
      writer.println(buffer.mkString)
    }
    println(s"Flushing ${_count} lines")
    writer.flush()
    
    writer.close()
    println("Stop")
  }
  
  private def isNewline(line:String):Boolean = {
    Try({
        val splits = line.split(",")
        //itemID,categoryID
        val i1 = if(splits(0).equals("itemID")) 0 else splits(0).toInt
        val i2 = if(splits(1).equals("categoryID")) 0 else splits(1).toInt
      })
      .isSuccess    
  }
 
  def main(args:Array[String]):Unit = {
    val inFile = if(args.length > 0) args(0) 
                 else "/home/kapil/Rough/Kaggle/Avito/ItemInfo_train.csv"
    val outFile = if(args.length > 1) args(1) 
                  else "/home/kapil/Rough/Kaggle/Avito/ItemInfo_train_clean.csv"
    val batch = if(args.length > 2) args(2).toInt else 10000
    
    cleanMultilineCsv(inFile, outFile, batch)
  }
}