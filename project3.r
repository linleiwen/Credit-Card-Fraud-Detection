rm(list=ls());
library(randomForest)
library(caret)
library(ROCR)
library(DMwR)
library(data.table)
library(zoo)
library(parallel)
detectCores()
setwd("C:/Users/leiwen/Desktop")
df <- fread("creditcard.csv")

#### Exploratory analysis
prop.table(table(df$Class))
summary(df)
sum(is.na(df)) ##check na

set.seed(1003)

ggplot(df, aes(x=V3)) + geom_density(aes(group=Class, colour=Class, fill=Class), alpha=0.3)

#### Data pre-processing
## 'normalize' the data
transform_columns <- c("V","Amount")
transformed_column     <- df[ ,grepl(paste(transform_columns, collapse = "|"),names(df)),with = FALSE]
transformed_column_processed <- predict(preProcess(transformed_column, method = c("BoxCox","scale")),transformed_column)

df_new <- data.table(cbind(transformed_column_processed,Class = df$Class))

df_new[,Class:=as.factor(Class)]

set.seed(1003)

#### split into Training and Test dataset
training_index <- createDataPartition(df_new$Class, p=0.7,list=FALSE)
training <- df_new[training_index,]
test<- df_new[-training_index,]

### Logistic regression
logit <- glm(Class ~ ., data = training, family = "binomial")
logit_pred <- predict(logit, test, type = "response")

logit_prediction <- prediction(logit_pred,test$Class)
logit_recall <- performance(logit_prediction,"prec","rec") ##precision vs recall
logit_roc <- performance(logit_prediction,"tpr","fpr") ## TP rate vs NP rate
logit_auc <- performance(logit_prediction,"auc")

### Random forest
rf.model <- randomForest(Class ~ ., data = training,ntree = 200, nodesize = 20)
rf_pred <- predict(rf.model, test,type="prob")

rf_prediction <- prediction(rf_pred[,2],test$Class)
rf_recall <- performance(rf_prediction,"prec","rec")
rf_roc <- performance(rf_prediction,"tpr","fpr")
rf_auc <- performance(rf_prediction,"auc")

### Bagging Trees
ctrl <- trainControl(method = "cv", number = 10)

tb_model <- train(Class ~ ., data = training, method = "treebag",
                 trControl = ctrl)

tb_pred <- predict(tb_model$finalModel, test, type = "prob")

tb_prediction <- prediction(tb_pred[,2],test$Class)
tb_recall <- performance(logit_prediction,"prec","rec")
tb_roc <- performance(logit_prediction,"tpr","fpr")
tb_auc <- performance(logit_prediction,"auc")


## xgboost
library(dplyr)
library(xgboost)
training[,Class:=as.integer(Class)-1]
test[,Class:=as.integer(Class)-1]

classifier = xgboost(data = as.matrix(training[,-30]), label = training$Class, nrounds = 100)
## as.matrix: xgboost only accept matrix;nrounds: train iteration 
# Predicting the Test set results
xgb_pred = predict(classifier, newdata = as.matrix(test[,-30]))

xgb_prediction <- prediction(xgb_pred,test$Class)
xgb_recall <- performance(xgb_prediction,"prec","rec") ##precision vs recall
xgb_roc <- performance(xgb_prediction,"tpr","fpr") ## TP rate vs NP rate
xgb_auc <- performance(xgb_prediction,"auc")


##plot result

plot(logit_recall,col='pink')
plot(rf_recall, add = TRUE, col = 'red')
plot(tb_recall, add = TRUE, col = 'green')
plot(xgb_recall, add = TRUE, col = 'black')


#### Functions to calculate 'area under the pr curve'

auprc <- function(pr_curve) {
  x <- as.numeric(unlist(pr_curve@x.values))
  y <- as.numeric(unlist(pr_curve@y.values))
  y[is.nan(y)] <- 1
  id <- order(x)
  result <- sum(diff(x[id])*rollmean(y[id],2))
  return(result)
}

auprc_results <- data.frame(logit=auprc(logit_recall)
                            , rf = auprc(rf_recall)
                            , tb = auprc(tb_recall)
                            , xgb = auprc(xgb_recall))
                            