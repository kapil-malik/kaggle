package com.kmalik.kaggle.avitoads

import com.kmalik.kaggle.utils.Utils
import com.kmalik.kaggle.utils.DfUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel

object DataPreparation {
  
  def main(args: Array[String]):Unit = {
    val sc = new SparkContext()
    val sqc = new SQLContext(sc)
    
    val pairsFile = Utils.readArg(args, "pairs")
    val infoFile = Utils.readArg(args, "info")
    val catFile = Utils.readArg(args, "category")
    val locFile = Utils.readArg(args, "location")
    val outFile = Utils.readArg(args, "output")
    
    val pairsDf = broadcast(csv(sqc, pairsFile))
    val infoDf = csv(sqc, infoFile)    
    val catDf = broadcast(csv(sqc, catFile))
    val locDf = broadcast(csv(sqc, locFile))
    
    pairsDf.registerTempTable("pairs")
    infoDf.registerTempTable("info")
    catDf.registerTempTable("cat")
    locDf.registerTempTable("loc")
    
    val p2 = sqc.sql("select p.*, "+
    "i.itemID as i_itemID_1, i.categoryID as categoryID_1, i.title as title_1, "+
    "i.description as description_1, i.images_array as images_array_1, i.attrsJSON as attrsJSON_1, "+
    "i.price as price_1, i.locationID as locationID_1, i.metroID as metroID_1, "+
    "i.lat as lat_1, i.lon as lon_1 from pairs p join info i on p.itemID_1 = i.itemID")
    p2.registerTempTable("pairs2")

    val p3 = sqc.sql("select p.*, "+
    "i.itemID as i_itemID_2, i.categoryID as categoryID_2, i.title as title_2, "+
    "i.description as description_2, i.images_array as images_array_2, i.attrsJSON as attrsJSON_2, "+
    "i.price as price_2, i.locationID as locationID_2, i.metroID as metroID_2, "+
    "i.lat as lat_2, i.lon as lon_2 from pairs2 p join info i on p.itemID_2 = i.itemID")
    p3.registerTempTable("pairs3")
    
    val p4 = sqc.sql("select p.*, l.regionID as regionID_1 from pairs3 p join loc l on p.locationID_1 = l.locationID")
    p4.registerTempTable("pairs4")
    
    val p5 = sqc.sql("select p.*, l.regionID as regionID_2 from pairs4 p join loc l on p.locationID_2 = l.locationID")
    p5.registerTempTable("pairs5")
    
    val p6 = sqc.sql("select p.*, c.parentCategoryID as parentCategoryID_1 from pairs5 p join cat c on p.categoryID_1 = c.categoryID")
    p6.registerTempTable("pairs6")
    
    val p7 = sqc.sql("select p.*, c.parentCategoryID as parentCategoryID_2 from pairs6 p join cat c on p.categoryID_2 = c.categoryID")
    p7.registerTempTable("pairs7")
    
    DfUtils.save(p7, outFile, "com.databricks.spark.csv", 128, Map("header"->"true"))
  }
  
  private def csv(sqc:SQLContext, path:String) = {
    DfUtils.load(sqc, path, "com.databricks.spark.csv", 
        Map("inferSchema"->"true","header"->"true"))
  }
}