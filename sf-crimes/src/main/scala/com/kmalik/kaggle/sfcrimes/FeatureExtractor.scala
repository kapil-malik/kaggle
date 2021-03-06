package com.kmalik.kaggle.sfcrimes

import java.text.SimpleDateFormat
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.attribute.NumericAttribute

object FeatureExtractor extends Serializable {
  val featureNamesAndTypes = Array(
	    ("districtNum", true),
	    ("dayOfWeekNum", true),
	    ("xCoord", false),
	    ("yCoord", false),
	    ("dayOfMonth", true),
	    ("month", true),
	    ("hr", true),
	    ("quarter", true),
	    ("weekend", true)
  )
  
  private val ID = "Id"
  private val DATES = "Dates"
  private val CATEGORY = "Category"
  private val DESCRIPT = "Descript"
  private val DAY_OF_WEEK = "DayOfWeek"
  private val PD_DISTRICT = "PdDistrict"
  private val RESOLUTION = "Resolution"
  private val ADDRESS = "Address"
  private val X_COORD = "X"
  private val Y_COORD = "Y"  
  
  private val DAY_OF_MONTH  = "DayOfMonth"
  private val MONTH         = "Month"
  private val HR            = "Hr"
  private val QUARTER       = "Quarter"  
}

class FeatureExtractor(override val uid: String, private val header:String) extends Transformer {
    
  def this(header:String) = this(Identifiable.randomUID("fXt"), header)

  private val headerSplit = header.split(",")
  private val dateIndex = headerSplit.indexOf(FeatureExtractor.DATES)
  private val dowIndex = headerSplit.indexOf(FeatureExtractor.DAY_OF_WEEK)
  private val districtIndex = headerSplit.indexOf(FeatureExtractor.PD_DISTRICT)
  private val xIndex = headerSplit.indexOf(FeatureExtractor.X_COORD)
  private val yIndex = headerSplit.indexOf(FeatureExtractor.Y_COORD)
                    
  private def buildDateColumn(row:String, dIndex:Int, col:String):Double = {
    val DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    val dateStr = row.split(",")(dIndex)
    val date = DATE_FORMAT.parse(dateStr)
    col match {
      case FeatureExtractor.DAY_OF_MONTH => return date.getDate()
      case FeatureExtractor.MONTH => return date.getMonth()
      case FeatureExtractor.HR => return date.getHours()
      case FeatureExtractor.QUARTER => return date.getHours()/3
      case _ => return date.getDate()
    }
  }
  
  private def isWeekend(dow:String) = if (dow.equals("Saturday") || dow.equals("Sunday")) 1.0 else 0.0
  
  private val strColumnUdf = udf((row:String, index:Int) => row.split(",")(index))
  private val numColumnUdf = udf((row:String, index:Int) => row.split(",")(index).toDouble)
  private val dateColumnUdf = udf((row:String, dateIndex:Int, col:String) 
                    => buildDateColumn(row, dateIndex, col))
  private val weekendUdf = udf((row:String, dowIndex:Int) => isWeekend(row.split(",")(dowIndex)))
  
  private def isValid(schema: StructType) = {
    schema.fieldNames.contains("row") &&
    schema.fieldNames.contains("label")
  }
  
  private def nominal(name:String) = NominalAttribute.defaultAttr.withName(name).toStructField()
  private def numeric(name:String) = NumericAttribute.defaultAttr.withName(name).toStructField()
  
  override def transform(df: DataFrame): DataFrame = {
    if (isValid(df.schema)) {
      df.select(df("label"),            
    	      strColumnUdf(df("row"), lit(districtIndex)).as("district"),
    				strColumnUdf(df("row"), lit(dowIndex)).as("dayOfWeek"),
    				numColumnUdf(df("row"), lit(xIndex)).as("xCoord"),
    				numColumnUdf(df("row"), lit(yIndex)).as("yCoord"),
            dateColumnUdf(df("row"), lit(dateIndex), lit(FeatureExtractor.DAY_OF_MONTH)).as("dayOfMonth"),
            dateColumnUdf(df("row"), lit(dateIndex), lit(FeatureExtractor.MONTH)).as("month"),
            dateColumnUdf(df("row"), lit(dateIndex), lit(FeatureExtractor.HR)).as("hr"),
            dateColumnUdf(df("row"), lit(dateIndex), lit(FeatureExtractor.QUARTER)).as("quarter"),
            weekendUdf(df("row"), lit(dowIndex)).as("weekend")
           )
    } else {
      df
    }
  }
  
  override def copy(extra: ParamMap): FeatureExtractor = defaultCopy(extra)
  
  override def transformSchema(schema: StructType): StructType = {
    if (isValid(schema)) {
      val label = schema.fields.find(_.name.equals("label")).get
      val xFields = FeatureExtractor.featureNamesAndTypes.map(x => {
        val name = x._1
        val isNominal = x._2
        if (isNominal) nominal(name) else numeric(name)
      })
      val outFields = label +: xFields
      StructType(outFields)
    } else {
      schema
    }
  }
}