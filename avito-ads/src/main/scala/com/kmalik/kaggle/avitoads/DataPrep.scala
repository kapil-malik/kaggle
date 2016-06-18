package com.kmalik.kaggle.avitoads

import scala.io.Source
import java.io.File
import scala.collection.mutable.{ListBuffer => uList, Map => uMap}
import scala.util.Try
import java.io.PrintWriter

object DataPrep {

  private val FLUSH_SIZE = 10000
  private val MAX_OPEN_LINES = 100
  
  def cleanMultilineCsv(inputPath:String, outputPath:String, 
    flushSize:Int = FLUSH_SIZE, maxOpenLines:Int = MAX_OPEN_LINES) = {
    println("Start")
    val reader = Source.fromFile(new File(inputPath)).getLines()
    val writer = new PrintWriter(new File(outputPath))
    val header = reader.next()
    
    writer.println(header)
    
    val clt = new CrossLineTracker(header, writer, flushSize, maxOpenLines)
    
    while(reader.hasNext) {
      clt.next

      val ilt = new InLineTracker(reader.next())
      
      while (ilt._index < ilt.length) {
        val nextQuoteIndex = ilt.line.indexOf("\"", ilt._index)
        val nextCommaIndex = ilt.line.indexOf(",", ilt._index)
                
        if ((nextCommaIndex == -1) || // No more commas till EOL OR
            (clt.isOpen && nextQuoteIndex == -1)) { // Open column till EOL                    
          handleEOLColumn(ilt, clt, nextQuoteIndex)          
        } else { // Parse a regular column                    
          if (!clt.isOpen) {            
            handleNewColumn(ilt, clt, nextCommaIndex, nextQuoteIndex)            
          } else {            
            handleOpenColumn(ilt, clt, nextQuoteIndex)
          }          
        }                
      } // EOL      
    } // EOF
    
    clt.close
    
    println("Stop")
  }

  private def handleEOLColumn(ilt:InLineTracker, clt:CrossLineTracker, quoteIndex:Int) = {
    val pending = ilt.line.substring(ilt._index)    
    ilt._index = ilt.length
    
    if (!clt.isOpen && !pending.startsWith("\"")) { // It's an unquoted column, at EOL                 
      clt.addCol(pending)
      clt.writeLine
      
    } else { // Continuing from an open column
      clt.openCol(pending)

      val nextQuoteIndex = if (pending.startsWith("\"")) ilt.line.indexOf("\"", quoteIndex + 1) else quoteIndex
      
      if (nextQuoteIndex == ilt.length -1) { // Open column closed itself, at EOL
        clt.closeCol()
        clt.writeLine
        
      } else if (nextQuoteIndex == -1) { // it's still open, continue        
        
      } else { // Quotes in middle of column not supported for now
        throw new RuntimeException(
          s"Quote pos ${nextQuoteIndex} found in middle of column pos ${ilt._index} at line ${clt.count}\n${ilt.line}")        
      }           
    }
    
  }

  private def handleNewColumn(ilt:InLineTracker, clt:CrossLineTracker, nextCommaIndex:Int, nextQuoteIndex:Int) = {
    if (nextCommaIndex < nextQuoteIndex || nextQuoteIndex == -1) { // Reading unquoted column
      
      clt.addCol(ilt.line.substring(ilt._index, nextCommaIndex))                             
      ilt._index = nextCommaIndex + 1
      
    } else { // Reading column with quotes              
      
      if (nextQuoteIndex != ilt._index) { // Quotes in middle of column not supported for now
        throw new RuntimeException(
          s"Quote pos ${nextQuoteIndex} found in middle of column pos ${ilt._index} at line ${clt.count}\n${ilt.line}")        
      }
                      
      val endQuoteIndex = ilt.line.indexOf("\"", nextQuoteIndex + 1)

      handleOpenColumn(ilt, clt, endQuoteIndex) // handle it as an open column
    }
  }
  
  private def handleOpenColumn(ilt:InLineTracker, clt:CrossLineTracker, nextQuoteIndex:Int) = {    
    if (nextQuoteIndex == -1) { // Quote column open till EOL
    
      clt.openCol(ilt.line.substring(ilt._index)) 
      ilt._index = ilt.length       
      
    } else if (nextQuoteIndex == ilt.length - 1){ // Quote column closed at EOL
      
      clt.closeCol(ilt.line.substring(ilt._index))
      clt.writeLine
      ilt._index = ilt.length       
      
    } else { // quote column closed midline
      
      val commaAfterQuoteIndex = ilt.line.indexOf(",", nextQuoteIndex + 1)
      
      if (commaAfterQuoteIndex != nextQuoteIndex + 1) { // Quotes in middle of column not supported for now          
        throw new RuntimeException(
          s"Quote pos ${nextQuoteIndex} found in middle of column pos ${ilt._index} at line ${clt.count}\n${ilt.line}")        
      }            
      
      clt.closeCol(ilt.line.substring(ilt._index, commaAfterQuoteIndex))
      ilt._index = commaAfterQuoteIndex + 1                              
    }
  }

  class CrossLineTracker(
    private val header:String,
    private val writer:PrintWriter,
    private val flushSize:Int = FLUSH_SIZE, 
    private val maxOpenLines:Int = MAX_OPEN_LINES
    ) {
    private val columns = header.split(",")
    private val ncol = columns.size
    
    private var _count = 0L
    private var _currCount = 0L
    
    private var _isOpen = false
    private var _openLinesCount = 0
    private var _openCol = ""
    private val cols = uList[String]()

    def isOpen = _isOpen
    
    def next = {
      _count += 1
      _currCount += 1
    }
    
    def count = _count
    
    def addCol(str:String) = {
      cols += str
      
      if (cols.size > ncol) {
        throw new RuntimeException(s"Added ${cols.size} columns, only ${ncol} expected. Line ${_count}")
      }
    }
    
    def openCol(str:String) = {
      _isOpen = true
      _openCol += str
      _openLinesCount += 1
      
      if (_openLinesCount >= maxOpenLines) {
        throw new RuntimeException(s"${_openLinesCount} lines parsed as open. Current line ${_count}")
      }
    }
    
    def closeCol(str:String = "") = {
      _openCol += str
      addCol(_openCol)
      
      _isOpen = false
      _openCol = ""
      _openLinesCount = 0
    }
    
    def writeLine = {
      if (cols.size == ncol) {
        // completed ncol columnns at EOL, write as line
        writer.println(cols.mkString(","))
        cols.clear()
        
        if (_currCount >= flushSize) {
          // parsed through more than $batch lines, flush to disk
          println(s"Flushing ${_currCount} lines")
          writer.flush()
          _currCount = 0
        }
      } else {
        throw new RuntimeException(s"Expected ${ncol} columns to be completed, found ${cols.size} at line ${_count}")
      }
    }
    
    def close = {
      if (!cols.isEmpty) {
        // unwritten columns available at EOF
        throw new RuntimeException(s"${cols.size} columns ${if(_isOpen) "(+open)" else ""} left unwritten")
      }
      println(s"Flushing ${_currCount} lines")
      writer.flush()
      
      writer.close()
    }
  }
  
  class InLineTracker(_line:String) {
    val line = _line.replaceAll("\"\"", "")
    val length = line.length()
    
    var _index = 0
  }
  
  def main(args:Array[String]):Unit = {
    val inFile = if(args.length > 0) args(0) 
                 else "/home/kapil/Rough/Kaggle/Avito/ItemInfo_train.csv"
    val outFile = if(args.length > 1) args(1) 
                  else "/home/kapil/Rough/Kaggle/Avito/ItemInfo_train_clean.csv"
    val flushSize = if(args.length > 2) args(2).toInt else 1000000
    val maxOpenLines = if(args.length > 3) args(3).toInt else 1000
    
    cleanMultilineCsv(inFile, outFile, flushSize, maxOpenLines)
  }
}