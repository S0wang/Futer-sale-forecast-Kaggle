# load library
library("xts")
library(TSA)
library(gmodels)
library(ggplot2)
library(urca)
library(dplyr)
library(plyr)
#library(tidyr)
library(forecast)
library(TSA)
library(ggfortify)
library(igraph)
library(lubridate)

# read-in datafiles
setwd("/Users/katewang/Desktop/2021 Fall/STA 560/project/predict-future-sales")
train_org = read.csv("/Users/katewang/Desktop/2021 Fall/STA 560/project/predict-future-sales/sales_train.csv")
test_org  = read.csv("/Users/katewang/Desktop/2021 Fall/STA 560/project/predict-future-sales/test.csv")
shop = read.csv("/Users/katewang/Desktop/2021 Fall/STA 560/project/predict-future-sales/shops.csv")

head(train_org)
# pre process data
# extract year and month
train_org$date_n = as.Date(train_org$date, "%d.%m.%Y")
train_org$month  =  as.numeric(format(train_org$date_n, "%m"))  #as.Date(cut(train_org$date_n, "month"))
train_org$year  = as.numeric(format(train_org$date_n, "%Y")) 
train_org$shop_id[train_org$shop_id==0] = 57
train_org$shop_id[train_org$shop_id==1] = 58
train_org$shop_id[train_org$shop_id==10] = 11

# standardize price
train_org$item_price_sd  = scale(train_org$item_price) 

# aggregate to monthly sales
train_org = train_org %>% select(year, month,shop_id, item_id, date_block_num, item_cnt_day, item_price,item_price_sd) %>% filter(item_cnt_day>=0&item_cnt_day<1000&item_price>0&item_price<100000) 
train     = train_org %>% dplyr::group_by(date_block_num,year,month,shop_id, item_id) %>% 
  dplyr::summarize(month_sale = sum(item_cnt_day),
                   month_price = median(item_price),
                   month_price_sd = median(item_price_sd))


# create a structure for each item in each shop for all date blocks
item_shop = unique(train[,c('shop_id','item_id')])
dim(item_shop)
item_shop2 = rep(item_shop, each = 34)
date_num = rep(seq(0,33), each = nrow(item_shop))
data = as.data.frame(cbind(date_num,item_shop2$shop_id,item_shop2$item_id))
colnames(data) = c('date_block_num','shop_id','item_id')

date_block = data.frame(date_block_num=seq(0,33),
                        year=c(rep(2013,12),rep(2014,12),rep(2015,10)),
                        month=c(rep(seq(1,12),2),seq(1,10)))
data  = data %>% left_join(date_block) 
data2 = data %>% left_join(train) %>% arrange(shop_id,item_id,date_block_num)
head(data2)
data2$month_sale[is.na(data2$month_sale)]=0
processed_all = data2 %>% fill(month_price, .direction = "updown")
processed_all = processed_all %>% fill(month_price_sd, .direction = "updown")

# export data
write.csv(processed_all,'/Users/katewang/Desktop/2021 Fall/STA 560/project/predict-future-sales/processed_all.csv', row.names=FALSE)




