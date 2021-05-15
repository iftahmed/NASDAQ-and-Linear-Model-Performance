

library(readxl)
library(tidyverse); library(modelr); ## packages for data manipulation and computing rmse easily.
library(glmnet)
library(glmnetUtils)
library(gridExtra)
library(MASS)
library(randomForest)
library(dplyr)

rm(list=ls())
data <- read_excel("Graduate/2021 Spring/STA 9890/Project/Processed_NASDAQ.xlsx")
data <- na.omit(data)

#Split data into X and y
y = data['Close']
X =   subset(data, select = -Close)

#Create n row for training and test data
n = nrow(data)
n.train        =     floor(0.8*n)
n.test         =     n-n.train

#M = # of times to run loop 
M              =     1
Rsq.test.ls    =     rep(0,M)  # ls = lasso
Rsq.train.ls   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  #en = elastic net
Rsq.train.en   =     rep(0,M)
Rsq.test.rid    =     rep(0,M)  #en = elastic net
Rsq.train.rid   =     rep(0,M)
Rsq.test.rf    =     rep(0,M)  #rf = random Forest
Rsq.train.rf   =     rep(0,M)

for (m in c(1:M)) {
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
  a=0 # elastic-net 0<a<1
  cv.fit           =     cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = TRUE, alpha = a, lambda = cv.fit$lambda.min)
  rid.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  rid.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net 0<a<1
  cv.fit           =     cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = TRUE, alpha = a, lambda = cv.fit$lambda.min)
  en.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  en.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)  
  
  # fit lasso and calculate and record the train and test R squares 
  a=1 # lasso
  cv.fit           =     cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = TRUE, alpha = a, lambda = cv.fit$lambda.min)
  las.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  las.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.ls[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.ls[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)   
  
  p=dim(data)[2]-1
  rf  =  randomForest(Close~., data=data, subset = train, mtry = floor(sqrt(p)), importance=TRUE)
  rf.train.hat   = predict(rf, newdata = X.train)
  rf.test.hat   = predict(rf, newdata = X.test)
  Rsq.test.rf[m] = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rf[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)   
  
  print(m)
}

#Create Train and Test Boxplots
library(ggplot2)
library(reshape)

training_rsqaured =  data.frame(Rsq.train.ls, Rsq.train.en, Rsq.train.rid, Rsq.train.rf)
test_rsquared = data.frame(Rsq.test.ls, Rsq.test.en, Rsq.test.rid, Rsq.test.rf   )

mdata_train <- melt(training_rsqaured)
mdata_test <- melt(test_rsquared)

p = ggplot(mdata_train, aes(x = variable, y= value)) +
    geom_boxplot() + 
    labs(title = 'Training Data R Sqaured', y = 'R Squared', x = 'Model')
p 
  
p = ggplot(mdata_test, aes(x = variable, y= value)) +
  geom_boxplot() + 
  labs(title = 'Testing Data R Sqaured', y = 'R Squared', x = 'Model')
p 
    
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
cv.rid = cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
plot(cv.rid, main = 'CV Curve for Ridge')

# fit elastic-net and calculate and record the train and test R squares 
a=0.5 # elastic-net 0<a<1
time_en = system.time(cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10))
cv.en = cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)
plot(cv.en, main = 'CV Curve for Elastic Net')

# fit lasso and calculate and record the train and test R squares 
a=1 # lasso
time_las = system.time(cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10))
cv.las = cv.glmnet(X.train, y.train, intercept = TRUE, alpha = a, nfolds = 10)  
plot(cv.las, main = 'CV Curve for Lasso')   

#fit random forest
p=dim(data)[2]-1
time_rf = system.time(randomForest(Close~., data=data,subset = train, mtry= floor(sqrt(p)), importance=TRUE))
rf <- randomForest(Close~., data=data,subset = train, mtry= floor(sqrt(p)), importance=TRUE)

#create 90% CI Rsq


t.test(Rsq.test.ls, conf.level = 0.9)
t.test(Rsq.test.en, conf.level = 0.9)
t.test(Rsq.test.rid, conf.level = 0.9)
t.test(Rsq.test.rf, conf.level = 0.9)


#Plot Coef


#lasso coefficients
lasso_coef = predict(cv.las, type = "coefficients")
lasso_coef <- as.data.frame(as.matrix(lasso_coef))

lasso_coef <- cbind(predictor = rownames(lasso_coef), lasso_coef)
rownames(lasso_coef) <- 1:nrow(lasso_coef)
colnames(lasso_coef) <- c('predictor','value')

lasso_coef = filter(lasso_coef, value != 0) %>% arrange()
lasso_coef$value = abs(lasso_coef$value)
lasso_coef <- top_n(lasso_coef, 10)

las_plot = ggplot(lasso_coef, aes(x = predictor, y = value)) +
  geom_bar(stat = 'identity') 
  




#ridge coef
coef = predict(cv.rid, type = "coefficients")
coef <- as.data.frame(as.matrix(coef))

coef <- cbind(predictor = rownames(coef), coef)
rownames(coef) <- 1:nrow(coef)
colnames(coef) <- c('predictor','value')

coef = filter(coef, value != 0) %>% arrange()
coef$value = abs(coef$value)
coef <- top_n(coef, 10)

rid_plot = ggplot(coef, aes(x = predictor, y = value)) +
  geom_bar(stat = 'identity') 




#Elastic Net Coef
coef = predict(cv.en, type = "coefficients")
coef <- as.data.frame(as.matrix(coef))

coef <- cbind(predictor = rownames(coef), coef)
rownames(coef) <- 1:nrow(coef)
colnames(coef) <- c('predictor','value')

coef = filter(coef, value != 0) %>% arrange()
coef$value = abs(coef$value)
coef <- top_n(coef, 10)

en_plot = ggplot(coef, aes(x = predictor, y = value)) +
  geom_bar(stat = 'identity') 




#RF coef
x <- importance(rf)

coef <- as.data.frame(as.matrix(x))

coef <- cbind(predictor = rownames(coef), coef)
rownames(coef) <- 1:nrow(coef)
colnames(coef) <- c('predictor','value')
coef <- coef[1:2]

coef$value = abs(coef$value)

coef <- coef %>% arrange(desc(value))
coef <- top_n(coef, 10)

rf_plot = ggplot(coef, aes(x = predictor, y = value)) +
  geom_bar(stat = 'identity') 

grid.arrange(rf_plot + ggtitle('Random ForestCoefficients'),
             en_plot + ggtitle('Elastic Net Coefficients'),
             las_plot + ggtitle('Lasso Coefficients'),
             rid_plot + ggtitle('Ridge Coefficients'))


las_test_residual = y.test - las.test.hat
rid_test_residual = y.test - rid.test.hat
en_test_residual = y.test - en.test.hat
rf_test_residual = y.test - rf.test.hat
par(mfrow=c(2,4))

boxplot(las_test_residual, main = 'Test Lasso')
boxplot(rid_test_residual, main = 'Test Ridge' )
boxplot(en_test_residual, main = 'Test Elastic Net' )
boxplot(rf_test_residual, main = 'Test Random Forest' )

las_train_residual = y.train - las.train.hat
rid_train_residual = y.train - rid.train.hat
en_train_residual = y.train - en.train.hat
rf_train_residual = y.train - rf.train.hat
par(mfrow=c(1,3))

boxplot(las_train_residual, main = 'Train Lasso')
boxplot(rid_train_residual, main = 'Train Ridge' )
boxplot(en_train_residual, main = 'Train Elastic Net' )
boxplot(rf_train_residual, main = 'Train Random Forest' )
