#predict using just data.frames

require(data.table)
require(caret)
training <- fread("pml-training.csv",stringsAsFactors=TRUE)
testing <- fread("pml-testing.csv",stringsAsFactors=TRUE)

#training <- read.csv("pml-training.csv",header=TRUE)
#get columns which are na or empty
homogenous = apply(testing, 2, function(var) length(unique(var)) == 1)
#take only those that have acutal data
testing <- testing[,which(!homogenous),with=FALSE]
training <- training[,which(!homogenous),with=FALSE]

#get rid of other fields that should not impact our results (user, timestamps)
training[,user_name:=NULL]
training[,raw_timestamp_part_1 := NULL]
training[,raw_timestamp_part_2 := NULL]
training[,cvtd_timestamp := NULL]
training[,V1 := NULL]

#change all classes in dataset to numeric for later calculations except for classe
for(i in 1:(length(training)-1)) training[[i]] = as.numeric(training[[i]])

#now split the training dataset into 2 groups, 1 for training and the other for cross validation (70-30)
training_split <- createDataPartition(training$classe, p = .7, list = FALSE)
train_ds <- training[training_split[,1]]
train_cv <- training[-training_split[,1]]

#preprocess the smaller training sets so that we eliminate any variables that are almost entirely the same
#making sure to get rid of the classe variable (59th column)
set.seed(123)
preProc <- preProcess(training[,-(length(training)),with=FALSE], method = "pca", thresh=.99)
#turns out we really only need about 37 variables for 99% variance
train_ds_preProc <- predict(preProc, train_ds[,-54,with=FALSE], )
train_cv_preProc <- predict(preProc, train_cv[,-54,with=FALSE], )

#add classe variable back in train_preProc
# train_preProc$classe <- training$classe

#use a random forest analysis to predict classe variable. More costly, but using for better results
#also using buildt in cross validation method in caret package
#because I am using data.tables for speed, I must specify classe as a factor (data.frame would have read it in as so)
set.seed(23421)
train_ds_preProc$classe <- as.factor(train_ds$classe)
modFit1 <- train(classe ~., method = "rf", data = train_ds_preProc, trControl = trainControl(method = "cv",number = 4),
                 importance = TRUE)


