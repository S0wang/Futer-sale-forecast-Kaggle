# Future-sale-forecast-Kaggle
https://www.kaggle.com/c/competitive-data-science-predict-future-sales
Provided with daily historical sales data. 
● sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
● test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
● sample_submission.csv - a sample submission file in the correct format.
● items.csv - supplemental information about the items/products.
● item_categories.csv  - supplemental information about the items categories.
● shops.csv- supplemental information about the shops.


Objective: to forecast the total amount of products sold in every shop for the test set. 


XGBoost model and linear regression
Feature Selection:
1.Deterministic trend: Month: 1-12
2. AR components (lags of 1, 2, 3, 4, 5)
         - Total item sales per month per shop
         - Total item sales per month per item 
         - Total item sales per month per category 
         - Price of items per shop (monthly median)
3. External regressors: 
         - Shop category/ shop city
         - Item category/ item subcategory
         - Shop ID & item ID


