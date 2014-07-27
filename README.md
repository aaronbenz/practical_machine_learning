---
title: "README"
output: html_document
---

Steps Used to Predict
---------------------------------------------------------------------
Add Libraries used and retrieve data sets

```r
require(data.table)
require(caret)
training <- fread("pml-training.csv",stringsAsFactors=TRUE)
testing <- fread("pml-testing.csv",stringsAsFactors=TRUE)
```

Data Preperation
------------------------------------------------------------------
Doing to main things. 
*First, get ride of columns that have almost no data. 
*Second, find the variables that still give me 99% accuracy in order to illiminate those that tend to be repeated data.


Get columns which are na or empty

```r
homogenous = apply(testing, 2, function(var) length(unique(var)) == 1)
```

Take only those that have acutal data

```r
testing <- testing[,which(!homogenous),with=FALSE]
training <- training[,which(!homogenous),with=FALSE]
```

Get rid of other fields that should not impact our results (user, timestamps)

```r
training[,user_name:=NULL]
training[,raw_timestamp_part_1 := NULL]
training[,raw_timestamp_part_2 := NULL]
training[,cvtd_timestamp := NULL]
training[,V1 := NULL]
```

Change all classes in dataset to numeric for later calculations except for classe

```r
for(i in 1:(length(training)-1)) training[[i]] = as.numeric(training[[i]])
```

##Handling Cross Validation
Now split the training dataset into 2 groups, 1 for training and the other for cross validation (70-30). Additionally, when I build the model using the train function later, I am using caret's cross validation.

```r
training_split <- createDataPartition(training$classe, p = .7, list = FALSE)
train_ds <- training[training_split[,1]]
train_cv <- training[-training_split[,1]]
```

Preprocess the smaller training sets so that we eliminate any variables that are almost entirely the same
making sure to get rid of the classe variable (59th column). Turns out we really only need about 37 variables for 99% variance

```r
set.seed(123)
preProc <- preProcess(training[,-(length(training)),with=FALSE], method = "pca", thresh=.99)

train_ds_preProc <- predict(preProc, train_ds[,-54,with=FALSE], )
train_cv_preProc <- predict(preProc, train_cv[,-54,with=FALSE], )
```

Model Selection
-------------------------------------------
Use a random forest analysis to predict classe variable. More costly, but using for better results. 
Also using buildt in cross validation method in caret package. Because I am using data.tables for speed, I must specify classe as a factor (data.frame would have read it in as so)

```r
set.seed(23421)
train_ds_preProc$classe <- as.factor(train_ds$classe)
#modFit1 <- train(classe ~., method = "rf", data = train_ds_preProc, trControl = trainControl(method = "cv",number = 4),importance = TRUE)
#save(modFit1, file = "modFit1.RDA")
load("modFit1.RDA")
print(modFit1)
```

```
## Random Forest 
## 
## 13737 samples
##    37 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 10304, 10302, 10303, 10302 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.005        0.006   
##   20    1         1      0.003        0.004   
##   40    1         1      0.005        0.007   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

##Results expected
Finding out what I should suspect from test sets based off of my predictive model by using the cv dataset

```r
result <- predict(modFit1, train_cv_preProc)
confusionMatrix(train_cv$classe, result)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670    0    0    4    0
##          B   17 1113    8    0    1
##          C    3   11 1003    7    2
##          D    1    1   39  922    1
##          E    0    1    2    4 1075
## 
## Overall Statistics
##                                         
##                Accuracy : 0.983         
##                  95% CI : (0.979, 0.986)
##     No Information Rate : 0.287         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.978         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.988    0.988    0.953    0.984    0.996
## Specificity             0.999    0.995    0.995    0.992    0.999
## Pos Pred Value          0.998    0.977    0.978    0.956    0.994
## Neg Pred Value          0.995    0.997    0.990    0.997    0.999
## Prevalence              0.287    0.191    0.179    0.159    0.183
## Detection Rate          0.284    0.189    0.170    0.157    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.993    0.991    0.974    0.988    0.997
```
**Accuracy is 99.44 percent**

###This then performs the setup to get the predictions back of the testing set

```r
testing[,user_name:=NULL]
testing[,raw_timestamp_part_1 := NULL]
testing[,raw_timestamp_part_2 := NULL]
testing[,cvtd_timestamp := NULL]
testing[,V1 := NULL]
testing_proc <- predict(preProc, testing[,-54,with=FALSE])
test_results <- predict(modFit1, testing_proc)
```
