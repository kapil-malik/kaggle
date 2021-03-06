# Avito Ads
http://kaggle.com/c/avito-duplicate-ads-detection

# Usage

## Data Cleanup
Clean multiline csv
```
java -cp avito-ads.jar com.kmalik.kaggle.avitoads.DataCleanup 
--input=ItemInfo_train.csv 
--output=ItemInfo_train_clean.csv 
--flushSize=1000000 
--maxOpenLines=1000
```

## Master data preparation
Prepare single data file from pairs, info, cat, loc.

Header:
itemID_1,itemID_2,isDuplicate,generationMethod,i_itemID_1,categoryID_1,title_1,description_1,images_array_1,attrsJSON_1,price_1,locationID_1,metroID_1,lat_1,lon_1,i_itemID_2,categoryID_2,title_2,description_2,images_array_2,attrsJSON_2,price_2,locationID_2,metroID_2,lat_2,lon_2,regionID_1,regionID_2,parentCategoryID_1,parentCategoryID_2

```
spark-submit --class com.kmalik.kaggle.avitoads.DataPreparation avito-ads.jar 
--pairs=ItemPairs_train.csv 
--info=ItemInfo_train_clean.csv 
--category=Category.csv 
--location=Location.csv 
--output=TrainMasterData
```

## Sample RF model
Example of how to use RF for building a simple model.
Currently uses only category, parentCategory, location, region columns of item1, item2 for modeling
```
spark-submit --class com.kmalik.kaggle.avitoads.SampleRf avito-ads.jar 
--data=TrainMasterData 
--output=outDir 
--trainRatio=0.8 
--numTrees=10 
--maxDepth=5 
--maxBins=100 
--evaluationMetric=areaUnderROC 
--seed=13309
```

## Sample RF model with cross-validation
Allows for multiple comma-separated options on numTrees, maxDepth, maxBins and evaluates best fit over parameter grid. It also uses k-fold cross validation on the train split, and then evaluates over the test split. Execution time is proportional to number of options * k.

It currently uses only category, parentCategory, location, region columns of item1, item2 for modeling.

*Clone this class* for further exploration on feature engineering (adding more features - categorical, numeric, text, image etc.).
```
spark-submit --class com.kmalik.kaggle.avitoads.SampleRfCV avito-ads.jar 
--data=TrainMasterData 
--output=outDir 
--partitions=64 // input data partitions
--trainRatio=0.8 // train - vs - test split
--numFolds=5 // cross validation folds for train data
--numTreesOpts=10 // comma separated options for number of trees
--maxDepthOpts=5 // comma separated options for max depth
--maxBinsOpts=100 // comma separated options for max bins
--evaluationMetric=areaUnderROC 
--seed=13309
```
