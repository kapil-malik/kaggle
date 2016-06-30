package org.apache.spark.ml

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.DefaultParamsReadable
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.StructType

class CustomBinaryClassificationEvaluator (override val uid: String)
  extends BinaryClassificationEvaluator(uid) {

  def this() = this(Identifiable.randomUID("customBinEval"))
  
  override def evaluate(dataset: DataFrame): Double = {
    val schema = dataset.schema
    CustomBinaryClassificationEvaluator.checkColumnTypes(schema, $(rawPredictionCol), Seq(DoubleType, new VectorUDT))
    SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)

    // TODO: When dataset metadata has been implemented, check rawPredictionCol vector length = 2.
    val scoreAndLabels = dataset.select($(rawPredictionCol), $(labelCol))
      .map { 
        case Row(rawPrediction: Vector, label: Double) => (rawPrediction(1), label)
        case Row(rawPrediction: Double, label: Double) => (rawPrediction, label)
      }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val metric = $(metricName) match {
      case "areaUnderROC" => metrics.areaUnderROC()
      case "areaUnderPR" => metrics.areaUnderPR()
    }
    metrics.unpersist()
    metric
  }
}

object CustomBinaryClassificationEvaluator extends DefaultParamsReadable[BinaryClassificationEvaluator] {

   def checkColumnTypes(
       schema: StructType,
       colName: String,
       dataTypes: Seq[DataType],
       msg: String = ""): Unit = {
     val actualDataType = schema(colName).dataType
     val message = if (msg != null && msg.trim.length > 0) " " + msg else ""
     require(dataTypes.exists(actualDataType.equals),
       s"Column $colName must be of type equal to one of the following types: " +
         s"${dataTypes.mkString("[", ", ", "]")} but was actually of type $actualDataType.$message")
   }
}
