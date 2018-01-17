library(randomForest)
library(caret)
library(ROCR)
library(DMwR)# SMOTE more positive cases
library(data.table)
library(zoo)
library(parallel)
library(ggplot2)
library(dplyr)
detectCores()

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

####smote

'''table(training$Class)
training <- SMOTE(Class ~ ., training, perc.over = 57600, perc.under=100) ## inflate "1" by x percentage, reduce "0" as y percentage
prop.table(table(training$Class))
table(training$Class) '''

### Logistic regression
logit <- glm(Class ~ ., data = training, family = "binomial")
logit_pred <- predict(logit, test, type = "response")

logit_prediction <- prediction(logit_pred,test$Class)
logit_recall <- performance(logit_prediction,"prec","rec") ##precision vs recall
logit_roc <- performance(logit_prediction,"tpr","fpr") ## TP rate vs NP rate
logit_auc <- performance(logit_prediction,"auc")

##kernel SVM
library(e1071)
ksvm.model = svm(formula = Class ~ .,
                 data = training,
                 type = 'C-classification', ## for classification
                 kernel = 'radial',probability=TRUE)  ## Gaussian not linear
KSVM_pred = predict(ksvm.model, test, probability=TRUE)

KSVM_prediction = prediction(attr(KSVM_pred,"probabilities")[,2],test$Class)
KSVM_recall <- performance(KSVM_prediction,"prec","rec") ##precision vs recall
KSVM_roc <- performance(KSVM_prediction,"tpr","fpr") ## TP rate vs FP rate
KSVM_auc <- performance(KSVM_prediction,"auc")


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
plot(KSVM_recall, add = TRUE, col = 'blue')


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
                            , xgb = auprc(xgb_recall)
                            ,KSVM = auprc(KSVM_recall) )
non_smote_aucpre = auprc_results
#smote_aucpre = auprc_results 
non_smote_aucpre


aucroc_results <- data.frame(logit=as.numeric(attr(logit_auc,"y.values"))
                            , rf = as.numeric(attr(rf_auc,"y.values"))
                            , tb = as.numeric(attr(tb_auc,"y.values"))
                            , xgb = as.numeric(attr(xgb_auc,"y.values"))
                            ,KSVM = as.numeric(attr(KSVM_auc,"y.values")) )

#non_smote_aucroc = aucroc_results
smote_aucroc = aucroc_results

temp = t(data.frame(rbind(non_smote_aucpre,smote_aucpre,non_smote_aucroc,smote_aucroc),row.names = c("non_smote_aucpre","smote_aucpre","non_smote_aucroc","smote_aucroc")))
temp = melt(temp,varnames = c("model","type"))
ggplot(data = temp, aes(x = type, y = value, colour = model, group = model))+geom_line(size = 1) 

##ggplot plot ROC and precision and recall curve
### Logistic regression 
sscurves1 <- evalmod(scores = logit_pred, labels = test$Class)
autoplot(sscurves1) 

##kernel SVM 
sscurves2 <- evalmod(scores = attr(KSVM_pred,"probabilities")[,2], labels = test$Class)
autoplot(sscurves2) 
##rf
sscurves3 <- evalmod(scores = rf_pred[,2], labels = test$Class)
autoplot(sscurves3) 

##tb
sscurves4 <- evalmod(scores = tb_pred[,2], labels = test$Class)
autoplot(sscurves4)

##xgboost

sscurves5 <- evalmod(scores = xgb_pred, labels = test$Class)
autoplot(sscurves5)

x = list(scores = list(list(logit_pred,attr(KSVM_pred,"probabilities")[,2],rf_pred[,2],tb_pred[,2],xgb_pred)),labels = test$Class,  modnames = c("random","poor_er","good_er","excel","excel"), dsids = c(1,1,1,1,1))
mdat <- mmdata(x[["scores"]], x["labels"],modnames = c("Logistic regression ","kernel SVM ","random forest","tree baging","xgboost"))

## Generate an mscurve object that contains ROC and Precision-Recall curves
mscurves <- evalmod(mdat)
## ROC and Precision-Recall curves
autoplot(mscurves)

sum(rf_pred[,2]>=0.00172)
cm = table(test$Class, rf_pred[,2]>=0.05)

cm/sum(cm)
(cm[1,1]+cm[2,2])/sum(cm)

sum(test$Class)/sum(cm)

save.image("project3.Rdata")
#load("project3.Rdata")
