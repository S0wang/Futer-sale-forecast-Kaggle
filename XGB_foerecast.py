
import sys
!{sys.executable} -m pip install xgboost

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

hyper_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9], 
                'gamma': [0, 0.5, 1, 1.5, 2, 5], 
                'subsample': [0.6, 0.7, 0.8, 0.9, 1], 
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                'learning_rate': [0.01, 0.1, 0.2, 0.3],https://github.com/S0wang/Futer-sale-forecast-Kaggle
                'max_bin' : [256, 512, 1024],
               }

xgbr = XGBRegressor(seed = 9999, tree_method = "hist", objective ="reg:tweedie") 
xgbr2 = XGBRegressor(seed = 9999, tree_method = "hist") 
xgbr3 = XGBRegressor(seed = 9999, tree_method = "hist", objective ="count:poisson") 

clf = RandomizedSearchCV(estimator = xgbr, 
                   param_distributions = hyper_params,
                   n_iter = 4, #
                   scoring = 'neg_root_mean_squared_error',
                   cv = splits,
                   verbose=3)
clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", -clf.best_score_)

## Linear regression as a reference
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
yhat_val_lr = lr.predict(X_val).clip(0,20)
print('Validation RMSE:', mean_squared_error(y_val, yhat_val_lr, squared=False)) #Validation RMSE: 0.9645168655662141
yhat_test_lr = lr.predict(X_test).clip(0,20)

## XGB with tweedie dist. 
from xgboost import XGBRegressor

ts = time.time()

xgb = XGBRegressor(seed = 999, 
    tree_method = "hist", 
    subsample = 1,
    max_depth = 9,
    learning_rate = 0.1,
    gamma = 1,
    colsample_bytree = 0.6,
    max_bin=256,
    objective ="reg:tweedie"
    )

xgb.fit(
    X_train,y_train,
    eval_metric="rmse",
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True,
    early_stopping_rounds = 10
    )

print('Training took: {0}s'.format(time.time()-ts))
#yhat_val_xgb = xgb.predict(X_val).clip(0, 20)
yhat_val_xgb = xgb.predict(X_val).clip(0,20)

print('Valdation RMSE:', mean_squared_error(y_val, yhat_val_xgb, squared=False)) #Valdation RMSE: 0.9409594444278176
yhat_test_xgb = xgb.predict(X_test) #.clip(0, 20)

## XGB with Poisson dist. 
ts = time.time()

xgbpos = XGBRegressor(seed = 999, 
    tree_method = "hist", 
    subsample = 1,
    max_depth = 9,
    learning_rate = 0.1,
    gamma = 1,
    colsample_bytree = 0.6,
    max_bin=256,
    objective ="count:poisson"
    )

xgbpos.fit(
    X_train,y_train,
    eval_metric="rmse",
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True,
    early_stopping_rounds = 10
    )

print('Training took: {0}s'.format(time.time()-ts))
#yhat_val_xgb = xgb.predict(X_val).clip(0, 20)
yhat_val_xgbpos = xgbpos.predict(X_val).clip(0,20)
print('Valdation RMSE:', mean_squared_error(y_val, yhat_val_xgbpos, squared=False)) #Valdation RMSE: 0.9409594444278176
yhat_test_xgbpos = xgbpos.predict(X_test) #.clip(0, 20)

## XGB with regression by default.
ts = time.time()

xgblr = XGBRegressor(seed = 999, 
    tree_method = "hist", 
    subsample = 1,
    max_depth = 9,
    learning_rate = 0.1,
    gamma = 1,
    colsample_bytree = 0.6,
    max_bin=256
    ) 

xgblr.fit(
    X_train,y_train,
    eval_metric="rmse",
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True,
    early_stopping_rounds = 10
    )

print('Training took: {0}s'.format(time.time()-ts))
#yhat_val_xgb = xgb.predict(X_val).clip(0, 20)
yhat_val_xgblr = xgblr.predict(X_val).clip(0,20)
print('Valdation RMSE:', mean_squared_error(y_val, yhat_val_xgblr, squared=False)) #Valdation RMSE: 0.9409594444278176
yhat_test_xgblr = xgblr.predict(X_test) #.clip(0, 20)


#matrix = pd.DataFrame(np.vstack(matrix), columns=['date_block_num', 'shop_id', 'item_id'], dtype=np.int16)
#np.vstack((yhat_val_lr, yhat_val_xgb, yhat_val_xgbpos, yhat_val_xgblr)).T
matrix_org = pd.read_csv('/Users/katewang/Desktop/2021 Fall/STA 560/project/predict-future-sales/matrix.csv')
matrix_org = matrix_org[matrix_org.date_block_num>=5] 
matrix_org.reset_index(drop=True, inplace=True)
 
X_val_id = matrix_org[matrix_org.date_block_num==33][["shop_id","item_id"]].copy()
X_val_id['yhat_val_xgb']= np.vstack(yhat_val_xgb)
X_val_id['yhat_val_xgbpos']= np.vstack(yhat_val_xgbpos)
X_val_id['yhat_val_xgblr']= np.vstack(yhat_val_xgblr)
X_val_id['yhat_val_lr']= np.vstack(yhat_val_lr)
X_val_id['y_val']= np.vstack(y_val)
X_val_id
X_val_id.to_csv('/Users/katewang/Desktop/2021 Fall/STA 560/project/predict-future-sales/leaf_xgb.csv', index=False)


import pickle
pickle.dump(xgb, open("xgboost_tweedie.pickle.dat", "wb"))
#loaded_model = pickle.load(open("xgboost_base.pickle.dat", "rb"))


from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(xgb, (10,14)) #weight # of item used 
plot_features(xgb, (10,14), 'gain') # improve rmse
plot_features(xgb, (10,14), 'cover')


y_train_meta = matrix[matrix.date_block_num.isin([ 27, 28, 29, 30, 31, 32])].item_cnt_month
X_train_meta = [[],[]]
for block in [27, 28, 29, 30, 31, 32]:
    print('Block:', block)
    # X and y Train for blocks from 12 to block
    X_train_block = matrix[matrix.date_block_num < block].drop(['date_block_num', 'item_cnt_month'], axis=1)
    y_train_block = matrix[matrix.date_block_num < block].item_cnt_month
    # X and y Test for block
    X_val_block = matrix[matrix.date_block_num == block].drop(['date_block_num', 'item_cnt_month'], axis=1)
    #y_test_block = matrix[matrix.date_block_num == block].item_cnt_month
    
    # Fit first model 
    print(' LR fitting ...')
    lr.fit(X_train_block, y_train_block)
    print(' LR fitting ... done')
    # Append prediction results on X_val_block to X_train_meta (first column)
    X_train_meta[0] += list(lr.predict(X_val_block).clip(0, 20))
    
    # Fit second model
    print(' XGB fitting ...')
    xgb.fit(
        X_train_block, y_train_block,
        eval_metric="rmse",
        eval_set=[(X_train_block, y_train_block)],
        #eval_set=[(X_train_block, y_train_block), (X_val_block, y_test_block)],
        verbose=0,
        early_stopping_rounds = 10
    )
    print(' XGB fitting ... done')

    # Append prediction results on X_val_block to X_train_meta (second column)
    X_train_meta[1] += list(xgb.predict(X_val_block).clip(0, 20))
# Turn list into dataframe
X_train_meta = pd.DataFrame({'yhat_lr': X_train_meta[0], 'yhat_xgb': X_train_meta[1]})

### Stacking lr and all XGB models
## leaf
stacking = LinearRegression()
stacking.fit(X_train_meta, y_train_meta)

#Squared: If True returns MSE value, if False returns RMSE value.
yhat_train_meta = stacking.predict(X_train_meta).clip(0, 20)
print('Meta Training RMSE:', mean_squared_error(y_train_meta, yhat_train_meta, squared=False))
# Meta Training RMSE: 0.7959949995252207

yhat_val_meta = stacking.predict(np.vstack((yhat_val_lr,yhat_val_xgb )).T).clip(0, 20)
print('Meta Validation RMSE:', mean_squared_error(y_val, yhat_val_meta, squared=False))
# Meta Validation RMSE: 0.9313002364522425

yhat_test_meta = stacking.predict(np.vstack((yhat_test_lr, yhat_test_xgb)).T).clip(0, 20)

X_train_meta = [[],[],[],[]]
for block in [27, 28, 29, 30, 31, 32]:
    print('Block:', block)
    # X and y Train for blocks from 12 to block
    X_train_block = matrix[matrix.date_block_num < block].drop(['date_block_num', 'item_cnt_month'], axis=1)
    y_train_block = matrix[matrix.date_block_num < block].item_cnt_month
    # X and y Test for block
    X_val_block = matrix[matrix.date_block_num == block].drop(['date_block_num', 'item_cnt_month'], axis=1)
    #y_test_block = matrix[matrix.date_block_num == block].item_cnt_month
    
    # Fit first model 
    print(' LR fitting ...')
    lr.fit(X_train_block, y_train_block)
    print(' LR fitting ... done')
    # Append prediction results on X_val_block to X_train_meta (first column)
    X_train_meta[0] += list(lr.predict(X_val_block).clip(0, 20))
    
    # Fit second model
    print(' XGB fitting ...')
    xgb.fit(
        X_train_block, y_train_block,
        eval_metric="rmse",
        eval_set=[(X_train_block, y_train_block)],
        #eval_set=[(X_train_block, y_train_block), (X_val_block, y_test_block)],
        verbose=0,
        early_stopping_rounds = 10
    )
    print(' XGB fitting ... done')
    # Append prediction results on X_val_block to X_train_meta (first column)
    X_train_meta[1] += list(xgb.predict(X_val_block).clip(0, 20))
    
    
    # Fit third model
    print(' XGBlr fitting ...')
    xgbpos.fit(
        X_train_block, y_train_block,
        eval_metric="rmse",
        eval_set=[(X_train_block, y_train_block)],
        #eval_set=[(X_train_block, y_train_block), (X_val_block, y_test_block)],
        verbose=0,
        early_stopping_rounds = 10
    )
    print(' XGBlr fitting ... done')
    # Append prediction results on X_val_block to X_train_meta (first column)
    X_train_meta[2] += list(xgbpos.predict(X_val_block).clip(0, 20))
    
    
    # Fit fourth model
    print(' XGBpos fitting ...')
    xgblr.fit(
        X_train_block, y_train_block,
        eval_metric="rmse",
        eval_set=[(X_train_block, y_train_block)],
        #eval_set=[(X_train_block, y_train_block), (X_val_block, y_test_block)],
        verbose=0,
        early_stopping_rounds = 10
    )
    print(' XGBpos fitting ... done')
 
    # Append prediction results on X_val_block to X_train_meta (second column)
    X_train_meta[3] += list(xgblr.predict(X_val_block).clip(0, 20))
# Turn list into dataframe
X_train_meta = pd.DataFrame({'yhat_lr': X_train_meta[0], 'yhat_xgb': X_train_meta[1], 'yhat_xgbpos': X_train_meta[2], 'yhat_xgblr': X_train_meta[3]})


stacking = LinearRegression()
stacking.fit(X_train_meta, y_train_meta)

#Squared: If True returns MSE value, if False returns RMSE value.
yhat_train_meta = stacking.predict(X_train_meta).clip(0, 20)
print('Meta Training RMSE:', mean_squared_error(y_train_meta, yhat_train_meta, squared=False))
# Meta Training RMSE: 0.7959949995252207

yhat_val_meta = stacking.predict(np.vstack((yhat_val_lr,yhat_val_xgb, yhat_val_xgbpos, yhat_val_xgblr)).T).clip(0, 20)
print('Meta Validation RMSE:', mean_squared_error(y_val, yhat_val_meta, squared=False))
# Meta Validation RMSE: 0.9313002364522425

yhat_test_meta = stacking.predict(np.vstack((yhat_test_lr, yhat_test_xgb, yhat_test_xgbpos, yhat_test_xgblr)).T).clip(0, 20)

ts = time.time()

xgbsta = XGBRegressor(seed = 999, 
    tree_method = "hist", 
    subsample = 1,
    max_depth = 9,
    learning_rate = 0.1,
    gamma = 1,
    colsample_bytree = 0.6,
    max_bin=256
    )

xgb.fit(
    X_train,y_train,
    eval_metric="rmse",
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True,
    early_stopping_rounds = 10
    )

print('Training took: {0}s'.format(time.time()-ts))
#yhat_val_xgb = xgb.predict(X_val).clip(0, 20)
yhat_val_xgb = xgb.predict(X_val).clip(0,20)

print('Valdation RMSE:', mean_squared_error(y_val, yhat_val_xgb, squared=False)) #Valdation RMSE: 0.9409594444278176
yhat_test_xgb = xgb.predict(X_test) #.clip(0, 20)


