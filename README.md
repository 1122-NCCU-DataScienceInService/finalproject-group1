# [Group1] Customer Churn Prediction
Retension(客戶留存) 對於每一間企業的客戶關係管理 Customer Relationship Management (CRM) 都是十分重要的。

這個比率關乎企業在帶進行新客戶時，可以留下多少客戶。企業要留得住客戶，才有辦法長久發展。

在這個期末報告中，我們將使用 Kaggle Telco-customer-churn 的資料集，找出哪些客戶可能不再使用公司服務。  

對客戶流失前進行挽留的行銷活動，為企業留住客戶。

## Analytics Highlight
* 在這個專案中，我們實作了一個在客戶關係管理中，偵測哪些客戶會流失的模型
* 我們採用了預測能力非常高的 XGBoost 模型做預測
* 採用 ROC, AUC, Lift Analysis 以及 Null Model Analysis 來評估模型成效
* 在預測的結果中，我們可以針對最容易流失的客戶，讓企業對潛在流失戶做精準行銷挽留客戶

## Contributors
|組員|系級|學號|工作分配|
|-|-|-|-|
|陳品萱|資科碩一|112971018|團隊的中流砥柱，一個人打十個|
|林宴葶|資科碩一|112971022|團隊的中流砥柱，一個人打十個|
|傅國書|資科碩一|112971025|年紀最大的團員，負責模型的訓練及測試部份|
|張祐誠|資科碩一|112971013|分析流程與程式架構規劃、程式彙整、報告說明撰寫| 

## Quick start
```R
我們將資料處理的流程拆分在不同的 R 腳本中：
* 探索式分析:Rscript code/profiling.R
* 資料清理：Rscript code/clean.R
* 特徵工程：Rscript code/feature_engineering.R
* 模型訓練：Rscript code/Training.R

基本的 Workflow:
profiling -> clean -> feature_engineering -> Training
```

## Source code description

*Project detail was written in docs/presentation.Rmd and .html files.*

### docs
    * presentation.Rmd: our project presentation Rmarkdown
    * presentation.html: our project presentation
    * schema.xlsx: data schema of the dataset

### data
    * raw folder: raw data
    * clean folder: data after cleaning
    * feature folder: data for maching learning

### code
    * profiling.R -> Doing data profiling
    * clean.R -> Cleaning dataset
    * feature_engineering.R -> making feature set for training model
    * Training.R -> Training & evaluating the model

### model
    * weight of churn model

### results
    * feature_importance.csv: How importance of each columns for rediction.
    * profiling.html: data profiling
    * roc_train/test.png: roc chart of prediction
    * train/test_prediction.csv: outputs prediction
    * lift_data.csv: lift analysis of data
    * lift_chart.png: lift chart of each customer segmentation

### README.md

Introducion of the final project.

## References

### Packages we used
* readr
* VIM
* dplyr
* caret
* ROSE
* car
* ggplot2
* lattice
* xgboost
* pROC

### Data source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Analytics Target
* 什麼樣特徵的人容易 Churn?
* 誰會 Churn? 準確度多少?

### Churn Definition
Customers who left within the last month – the column is called Churn.  
