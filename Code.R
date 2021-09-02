

library(readxl)      #Reading in csv excel file
library(tidyverse)  
library(modelr)      #Data manipulation and computing rmse easily.
library(glmnet)      #Fits generalized linear models 
library(glmnetUtils) #Allows streamlining of fitting elastic net models with glmnet
library(gridExtra)   #Used for Grid graphics
library(MASS)
library(randomForest)#Used for classification and regression based on random forest
library(dplyr)

rm(list=ls())
setwd("~/Graduate/2021 Spring/STA 9890/Project")

data <- read_excel("Processed_NASDAQ.xlsx")
data <- na.omit(data) #The models used do not work if there are 'NA' in the data

#Split data into X (response) and y (predictor)
y = data['Close']  #Select just the column labeled 'Close', the daily closing price of the NASDAQ
X = subset(data, select = -Close) #Select all columns not labeled Close

#Create n row for training and test data
n = nrow(data)
n.train        =     floor(0.8*n)  #We use 80% of the data to train our models and the rest is used to test our models
n.test         =     n-n.train

#M = # of times to run loop  
M              =     10        # I would recommend using at least 100 iterations. 10 is used to lower time to run models

#I need to initialize the lists needed to store the R-sqaured values of the test and training data of the models
Rsq.test.ls    =     rep(0,M)  # ls = lasso
Rsq.train.ls   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  # en = elastic net
Rsq.train.en   =     rep(0,M)
Rsq.test.rid   =     rep(0,M)  # rid = ridge regression
Rsq.train.rid  =     rep(0,M)
Rsq.test.rf    =     rep(0,M)  # rf = random Forest
Rsq.train.rf   =     rep(0,M)

#This loop runs our 4 models 'M' number of times. We are trying to find the R-squared value of each model in each iteration  
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)                     #Create random order so each iteration is different 
  train            =     shuffled_indexes[1:n.train]   #Chose first 80% of the data for the training set 
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train,]
  X.test           =     X[test, ]
  y.test           =     y[test,]
  
  X.train = as.matrix(X.train)  #The models require the inputs to be in matrix form 
  y.train = as.matrix(y.train)
  X.test = as.matrix(X.test)
  y.test = as.matrix(y.test)
  
  # fit ridge and calculate and record the train and test R squares 
  a=0 # elastic-net 0<a<1
  cv.fit             =     cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
  fit                =     glmnet(X.train, y.train,intercept = TRUE, alpha = a, lambda = cv.fit$lambda.min)
  rid.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  rid.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]    =     1-mean((y.test - rid.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rid[m]   =     1-mean((y.train - rid.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net 0<a<1
  cv.fit            =     cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
  fit               =     glmnet(X.train, y.train,intercept = TRUE, alpha = a, lambda = cv.fit$lambda.min)
  en.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  en.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]    =     1-mean((y.test - en.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m]   =     1-mean((y.train - en.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  
  # fit lasso and calculate and record the train and test R squares 
  a=1 # lasso
  cv.fit            =     cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
  fit               =     glmnet(X.train, y.train,intercept = TRUE, alpha = a, lambda = cv.fit$lambda.min)
  las.train.hat     =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  las.test.hat      =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.ls[m]    =     1-mean((y.test - las.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.ls[m]   =     1-mean((y.train - las.train.hat)^2)/mean((y.train - mean(y.train))^2)   
  
  # fit random forest and calculate and record the train and test R squares
  p                 =     dim(data)[2]-1
  rf                =     randomForest(Close~., data=data, subset = train, mtry = floor(sqrt(p)), importance=TRUE)
  rf.train.hat      =     predict(rf, newdata = X.train)
  rf.test.hat       =     predict(rf, newdata = X.test)
  Rsq.test.rf[m]    =     1-mean((y.test - rf.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rf[m]   =     1-mean((y.train - rf.train.hat)^2)/mean((y.train - mean(y.train))^2)   
  
  print(m) 
}

#Create Train and Test Boxplots
library(ggplot2)
library(reshape)

training_rsqaured =  data.frame(Rsq.train.ls, Rsq.train.en, Rsq.train.rid, Rsq.train.rf)
test_rsquared = data.frame(Rsq.test.ls, Rsq.test.en, Rsq.test.rid, Rsq.test.rf   )

mdata_train <- melt(training_rsqaured)
mdata_test <- melt(test_rsquared)

#Create box plot of the training R-squared values of the different models 
plot_train = ggplot(mdata_train, aes(x = variable, y= value)) +
    geom_boxplot() + 
    labs(title = 'Training Data R Sqaured', y = 'R Squared', x = 'Model')
plot_train 

#Create box plot of the test R-squared values of the different models   
plot_test = ggplot(mdata_test, aes(x = variable, y= value)) +
  geom_boxplot() + 
  labs(title = 'Testing Data R Sqaured', y = 'R Squared', x = 'Model')
plot_test 
    
#Find time to run ridge, lasso, en 
shuffled_indexes =     sample(n)         #Create random order 
train            =     shuffled_indexes[1:n.train]
test             =     shuffled_indexes[(1+n.train):n]
X.train          =     X[train, ]
y.train          =     y[train,]
X.test           =     X[test, ]
y.test           =     y[test,]

X.train = as.matrix(X.train)
y.train = as.matrix(y.train)
X.test = as.matrix(X.test)
y.test = as.matrix(y.test)

# fit ridge and calculate and record the train and test R squares 
a=0 # ridge = 0
time_rid = system.time(cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10))
time_rid

#
cv.rid = cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
plot(cv.rid, main = 'CV Curve for Ridge')

# fit elastic-net and calculate and record the train and test R squares 
a=0.5 # elastic-net 0<a<1
time_en = system.time(cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10))
time_en
cv.en = cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
plot(cv.en, main = 'CV Curve for Elastic Net')

# fit lasso and calculate and record the train and test R squares 
a=1 # lasso
time_las = system.time(cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10))
time_las
cv.las = cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)  
plot(cv.las, main = 'CV Curve for Lasso')   

#fit random forest
p=dim(data)[2]-1
time_rf = system.time(randomForest(Close~., data=data,subset = train, mtry= floor(sqrt(p)), importance=TRUE))
time_rf
rf <- randomForest(Close~., data=data,subset = train, mtry= floor(sqrt(p)), importance=TRUE)


#create 90% CI for R-squared values
t.test(Rsq.test.ls, conf.level = 0.9)
t.test(Rsq.test.en, conf.level = 0.9)
t.test(Rsq.test.rid, conf.level = 0.9)
t.test(Rsq.test.rf, conf.level = 0.9)


