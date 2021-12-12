import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import time
from itertools import product
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import base
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

INPUTFOLDER = '../input/competitive-data-science-predict-future-sales/'

item_categories = pd.read_csv(os.path.join(INPUTFOLDER, 'item_categories.csv'))
items           = pd.read_csv(os.path.join(INPUTFOLDER, 'items.csv'))
sales           = pd.read_csv(os.path.join(INPUTFOLDER, 'sales_train.csv'))
shops           = pd.read_csv(os.path.join(INPUTFOLDER, 'shops.csv'))
test            = pd.read_csv(os.path.join(INPUTFOLDER, 'test.csv'))
              
# EXTRACTING THE YEAR AND THE MONTH FROM THE DATE COLUMN
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
LINEWIDTH=2
ALPHA=.6

data = sales[['date', 'date_block_num','item_cnt_day']].copy()

# Extract the year and the month from the date column into indepedent columns
data['date']  = pd.to_datetime(data['date'], format='%d.%m.%Y')
data['year']  = data['date'].dt.year
data['month'] = data['date'].dt.month
data.drop(['date'], axis=1, inplace=True)
print(data.head())

# sum the number of sold items for each date block number 
data = data.groupby('date_block_num', as_index=False)\
       .agg({'year':'first', 'month':'first', 'item_cnt_day':'sum'})\
       .rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=False)
print(data.head())


# filter outliers
sales = sales[(sales.item_price<100000)&(sales.item_price>0)]
sales = sales[(sales.item_cnt_day>0)&(sales.item_cnt_day<1000)]
print(sales)
# Remove duplicate shops
import collections
print(collections.Counter(sales.shop_id))
sales.loc[sales.shop_id==0, 'shop_id'] = 57
test.loc[test.shop_id==0, 'shop_id'] = 57

sales.loc[sales.shop_id==1, 'shop_id'] = 58
test.loc[test.shop_id==1, 'shop_id'] = 58

sales.loc[sales.shop_id==10, 'shop_id'] = 11
test.loc[test.shop_id==10, 'shop_id'] = 11

# Extract shop city and category from shop names
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name"] = 'СергиевПосад ТЦ "7Я"'
shops["shop_city"] = shops.shop_name.str.split(' ').map(lambda x: x[0])
shops["shop_category"] = shops.shop_name.str.split(" ").map(lambda x: x[1])
shops.loc[shops.shop_city == "!Якутск", "shop_city"] = "Якутск" 
print(shops.head())

shops["shop_city"] = LabelEncoder().fit_transform(shops.shop_city)
shops["shop_category"] = LabelEncoder().fit_transform(shops.shop_category)
shops = shops[["shop_id", "shop_category", "shop_city"]]
shops.head()

print(shops.shop_city)
print(LabelEncoder().fit_transform(shops.shop_city))

# Extract item city and category from shop names
item_categories["category_type"] = item_categories.item_category_name.apply(lambda x: x.split(" ")[0]).astype(str)
# The category_type "Gamming" and "accesoires" becomes "Games"
item_categories.loc[(item_categories.category_type=="Игровые")|(item_categories.category_type=="Аксессуары"), "category_type"] = "Игры"
item_categories["split"] = item_categories.item_category_name.apply(lambda x: x.split("-"))
item_categories["category_subtype"] = item_categories.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_categories["category_type"] = LabelEncoder().fit_transform(item_categories.category_type)
item_categories["category_subtype"] = LabelEncoder().fit_transform(item_categories.category_subtype)
item_categories = item_categories[["item_category_id", "category_type", "category_subtype"]]
item_categories.head()

# aggregate daily sales to monthly sales
print(sales.head())
sales = sales.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)\
          .agg({'item_cnt_day':'sum'})\
          .rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=False)
        
test['date_block_num'] = 34
test['item_cnt_month'] = 0
print(test.head())
del test['ID']

df = sales.append(test)
df

matrix = []
for num in df['date_block_num'].unique(): 
    tmp = df[df.date_block_num==num]
    matrix.append(np.array(list(product([num], tmp.shop_id.unique(), tmp.item_id.unique())), dtype='int16'))
print(matrix)

# Turn the grid into a dataframe
matrix = pd.DataFrame(np.vstack(matrix), columns=['date_block_num', 'shop_id', 'item_id'], dtype=np.int16)

# Add the features from sales data to the matrix
matrix = matrix.merge(df, how='left', on=['date_block_num', 'shop_id', 'item_id']).fillna(0)

#Merge features from shops, items and item_categories:
matrix = matrix.merge(shops, how='left', on='shop_id')
matrix = matrix.merge(items[['item_id','item_category_id']], how='left', on='item_id')
matrix = matrix.merge(item_categories, how='left', on='item_category_id')

# Add month
matrix['month'] = matrix.date_block_num%12
# Clip counts
matrix['item_cnt_month'] = matrix['item_cnt_month'].clip(0,20)#.clip(0, 20)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix['month'] = matrix['month'].astype(np.int8)
matrix['item_cnt_month'] = matrix['item_cnt_month'].astype(np.int32)
matrix['shop_category'] = matrix['shop_category'].astype(np.int8)
matrix['shop_city'] = matrix['shop_city'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['category_type'] = matrix['category_type'].astype(np.int8)
matrix['category_subtype'] = matrix['category_subtype'].astype(np.int8)
matrix

def lag_feature(df, lags, col):
    print(col)
    for i in lags:
        shifted = df[["date_block_num", "shop_id", "item_id", col]].copy()
        shifted.columns = ["date_block_num", "shop_id", "item_id", col+"_lag_"+str(i)]
        shifted.date_block_num += i
        df = df.merge(shifted, on=['date_block_num','shop_id','item_id'], how='left').fillna(0)
    return df
  
matrix = lag_feature(matrix, [1, 2, 3, 4, 5], 'item_cnt_month')
matrix

# cnt_block_shop
gb = matrix.groupby(['shop_id', 'date_block_num'],as_index=False)\
          .agg({'item_cnt_month':'sum'})\
          .rename(columns={'item_cnt_month':'cnt_block_shop'}, inplace=False)
matrix = matrix.merge(gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)
matrix = lag_feature(matrix, [1, 2, 3, 4, 5], 'cnt_block_shop')
matrix.drop('cnt_block_shop', axis=1, inplace=True)
# cnt_block_item
gb = matrix.groupby(['item_id', 'date_block_num'],as_index=False)\
          .agg({'item_cnt_month':'sum'})\
          .rename(columns={'item_cnt_month':'cnt_block_item'}, inplace=False)
matrix = matrix.merge(gb, how='left', on=['item_id', 'date_block_num']).fillna(0)
matrix = lag_feature(matrix, [1, 2, 3, 4, 5], 'cnt_block_item')
matrix.drop('cnt_block_item', axis=1, inplace=True)
# cnt_block_category
gb = matrix.groupby(['category_type', 'date_block_num'],as_index=False)\
          .agg({'item_cnt_month':'sum'})\
          .rename(columns={'item_cnt_month':'cnt_block_category'}, inplace=False)
matrix = matrix.merge(gb, how='left', on=['category_type', 'date_block_num']).fillna(0)
matrix = lag_feature(matrix, [1, 2, 3, 4, 5], 'cnt_block_category')
matrix.drop('cnt_block_category', axis=1, inplace=True)

#encode 
from sklearn.preprocessing import StandardScaler

def standard_mean_enc(df, col):
    mean_enc = df.groupby(col).agg({'item_cnt_month': 'mean'})
    scaler = StandardScaler().fit(mean_enc)
    return {v: k[0] for v, k in enumerate(scaler.transform(mean_enc))}

cols_to_mean_encode = ['shop_category', 'shop_city', 'item_category_id', 'category_type', 'category_subtype']


for col in cols_to_mean_encode:
    # Train on the train data
    mean_enc = standard_mean_enc(matrix[matrix.date_block_num < 33].copy(), col) # X_train, y_train
    # Apply to Train, Validation and Test
    matrix[col] = matrix[col].map(mean_enc)
matrix


# cnt_block_category
sales           = pd.read_csv(os.path.join(INPUTFOLDER, 'sales_train.csv'))
#sales = sales["item_id","item_price"].groupby('item_id').transform(lambda x: (x - x.mean()) / x.std())
groups = sales[["item_price","item_id"]].groupby("item_id")

mean, std = groups.transform("mean"), groups.transform("std")
#mean2, std2 = sales[["item_price"]].transform("mean"),  sales[["item_price"]].transform("std")
#sales[["mean"]] ,sales[["std"]] = groups.transform("mean"), groups.transform("std")

sales[["item_price_sd"]] = (sales[["item_price"]] - mean) / std
sales[["item_price_sd2"]] = sales[["item_price"]].transform(lambda x: (x - x.mean()) / x.std())
sales

def lag_feature(df, lags, col):
    print(col)
    for i in lags:
        shifted = df[["date_block_num", "shop_id", "item_id", col]].copy()
        shifted.columns = ["date_block_num", "shop_id", "item_id", col+"_lag_"+str(i)]
        shifted.date_block_num += i
        df = df.merge(shifted, on=['date_block_num','shop_id','item_id'], how='left').fillna(0)
    return df

gb = sales.groupby(['shop_id', "item_id", 'date_block_num'],as_index=False)\
          .agg({'item_price_sd2':'median'})\
          .rename(columns={'item_price_sd2':'price_date_block'}, inplace=False)
matrix = matrix.merge(gb, how='left', on=['shop_id', "item_id", 'date_block_num']).fillna(0)
matrix = lag_feature(matrix, [1, 2, 3, 4, 5], 'price_date_block')
matrix.drop('price_date_block', axis=1, inplace=True)

def standard_mean_enc(df, col):
    mean_enc = df.groupby(col).agg({'item_cnt_month': 'mean'})
    scaler = StandardScaler().fit(mean_enc)
    return {v: k[0] for v, k in enumerate(scaler.transform(mean_enc))}

cols_to_mean_encode = ['shop_id', 'item_id']

for col in cols_to_mean_encode:
    # Train on the train data
    mean_enc = standard_mean_enc(matrix[matrix.date_block_num < 33].copy(), col) # X_train, y_train
    # Apply to Train, Validation and Test
    matrix[col] = matrix[col].map(mean_enc)
matrix['shop_id']=matrix['shop_id'].fillna(0)
matrix['item_id']=matrix['item_id'].fillna(0)


matrix = matrix[matrix.date_block_num>=5] 
matrix.reset_index(drop=True, inplace=True)
matrix


X_train = matrix[matrix.date_block_num < 33].drop(['item_cnt_month'], axis=1)
y_train = matrix[matrix.date_block_num < 33]['item_cnt_month']
X_val = matrix[matrix.date_block_num == 33].drop(['item_cnt_month'], axis=1)
y_val =  matrix[matrix.date_block_num == 33]['item_cnt_month']
X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis=1)

X_train.drop('date_block_num', axis=1, inplace=True)
X_val.drop('date_block_num', axis=1, inplace=True)
X_test.drop('date_block_num', axis=1, inplace=True)




