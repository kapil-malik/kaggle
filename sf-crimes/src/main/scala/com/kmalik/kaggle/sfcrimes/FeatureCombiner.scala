package com.kmalik.kaggle.sfcrimes

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.StructType
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.WrappedArray
import org.apache.spark.sql.functions._
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.attribute.NumericAttribute
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.sql.types.StructField

object FeatureCombiner {
  private val featuresUdf = udf((features:WrappedArray[Double]) => Vectors.dense(features.toArray))
  
  private def nominal(name:String) = NominalAttribute.defaultAttr.withName(name)  
  private def numeric(name:String) = NumericAttribute.defaultAttr.withName(name)
}

class FeatureCombiner(override val uid: String, 
  private val ftNameTypes:Array[(String, Boolean)]) extends Transformer {
  
  def this(ftNameTypes:Array[(String, Boolean)]) = this(Identifiable.randomUID("fCmb"), ftNameTypes)
  def this(ftNames:Array[String]) = this(Identifiable.randomUID("fCmb"), ftNames.map(x => (x, true)))
  
  private val featuresField = buildFeaturesField()
  
  override def transform(df: DataFrame): DataFrame = {
    if (isValid(df.schema)) {
      val labelCol = df("label")
      val fColArray = ftNameTypes.map(_._1).map(x => df(x))
      val fArrayCol = array(fColArray:_*)
      val featuresCol = FeatureCombiner.featuresUdf(fArrayCol).as("features")
      df.select(labelCol, featuresCol)
    } else {
      df
    }
  }
  
  override def copy(extra: ParamMap): FeatureExtractor = defaultCopy(extra)
  
  override def transformSchema(schema: StructType): StructType = {
    if (isValid(schema)) {
      val labelField = schema.fields.find(_.name.equals("label")).get
      val fields = Array(labelField, featuresField)
      StructType(fields)
    } else {
      schema
    }
  }
  
  private def buildFeaturesField(): StructField = {
      val ftAttrs = ftNameTypes.map(x=> {
        val name = x._1
        val isNominal = x._2
        val fAttr = if(isNominal) FeatureCombiner.nominal(name) else FeatureCombiner.numeric(name) 
        fAttr.asInstanceOf[Attribute]
      })
      new AttributeGroup("features", ftAttrs).toStructField()
  }
  
  private def isValid(schema: StructType) = {
    val fields = schema.fieldNames
    fields.contains("label") && ftNameTypes.map(_._1).forall(fields.contains)
  }
}