# NASDAQ-and-Linear-Model-Performance

## Background 
The NASDAQ Composite is a stock market index that consists of the stocks on the NASDAQ stock exchange. It is one of the most followed stock indexes in the United States.
In this project, I will be comparing the model performance of different linear models. These models will try to predict NASDAQ closing prices by using over 40 different financial variables. These variables include commodities, treasury bills, currencies, futures, and stock prices of select large companies.

The linear models I will be using are Ridge Regression, Lasso Regression, Elastic Net Regression, and Random Forest. I will be comparing the performance of the different techniques using the R-squared values and the time it takes to calcualte each model. 

## Results

I found that Ridge Regression, Lasso Regression, Elastic Net Regression are similar in terms of the R-squared value and the time it takes to run each model. Random Forest has a 
higher R-squared value which means that this value better predicts the closing NASDAQ price change. However, it takes 50 times longer to run a Random Forest model versus the other
three models for our data.
