#!/usr/bin/env python
# coding: utf-8

# ## Solar_energy_forecast

# #### 1 -- Load_Libraries

# In[2]:


get_ipython().system(' pip install lightgbm')
get_ipython().system(' pip install xgboost')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error # MSE
from sklearn.metrics import mean_absolute_error # MAE
import numpy as np
from sklearn.metrics import mean_squared_log_error
import re
from sklearn.model_selection import cross_val_score


# #### 2 -- Functions

# In[4]:


# Scores computation
def score (a,b):
    # a = y_test, b = y_predictor
    R2 = r2_score(a,b)
    mse_rf = mean_squared_error(a,b)
    rmse_rf = mse_rf**0.5
    mae_rf =  mean_absolute_error(a,b)
    # round with 3 decimals
    print('R2 =', round(R2,3))
    print('MSE =', round(mse_rf,3))
    print('RMSE =', round(rmse_rf,3))
    print('MAE = ', round(mae_rf,3))


# In[5]:


# R2 adjusted
# a = y_test, b = y_predicted, c = x_test
def adjusted_r2(a,b,c):
    adj_r2 = (1 - ((1 - r2_score(a, b)) * (len(a) - 1)) / 
          (len(a) - c.shape[1] - 1))
    return round(adj_r2,3)


# In[6]:


# Comparison to the mean_train
def comp_to_mean(t,b,z,w):
    # t = y_train, b = y_test, z = mse_algorithm, w = mae_algorithm
    a = np.mean(t) # mean value of the y_train
    a = [a]*len(b) # array with a
    mse_m = mean_squared_error(b,a)
    mae_m = mean_absolute_error(b,a)
    # comparison between algorithm error and mse_mean_train
    mse_mean_ratio = z/mse_m
    mae_mean_ration = w/mae_m
    print('MSE_ytest & mean_ytrain=', round(mse_m,3))
    print('MSE_algorithm = ', round(z,3))
    print('Ratio_MSE_algorithm & MSE_ytest & mean_ytrain=',round(mse_mean_ratio,3))
    print('MAE_ytest & mean_ytrain=', round(mae_m,3))
    print('MAE_algorithm = ', round(w,3))
    print('Ratio_MAE_algorithm & MAE_ytest & mean_ytrain=',round(mae_mean_ration,3))


# In[7]:


def data_float(df):
    df.set_index("date_m", inplace = True)
    for a in df.columns:
        df[a] = df[a].astype(float)
    # df.reset_index(inplace = True)


# In[8]:


def data_count(df):
    #df.set_index("date_m", inplace = True)
    for a in df.columns:
        b = -999.0
        print(a,df[a].value_counts()[b])
    #df.reset_index(inplace = True)


# In[9]:


def data_replace(df):
    #df.set_index("date_m", inplace = True)
    for a in df.columns:
        z = df[a].median()
        print(a,z)
        b = -999.0
        df[a] = df[a].replace(b,z)
    #df.reset_index(inplace = True)


# In[10]:


def data_resample_W(df):
    df = df.resample('W').mean()
    df = df.round(decimals = 3)
    return df


# ## 3 -- Hyperparameters_grid

# In[11]:


# Light & XG Boosting
params_grid_g= {'booster':["gbtree","gblinear",'dart'],'max_depth':[3,4,5,8,10,11,12],
                'min_child_weigth':[1,3,5,7],'gamma':[0,0.1,0.2,0.3,0.4],
                'colsample_bytree':[0.3,0.4,0.5,0.6],'learning_rate':[0.05,0.1,0.2,0.3,0.4,0.5,1.0,1.1]}


# In[12]:


# Gradient Boosting
params0= {
    'min_samples_split': [300,400,500],
    'min_samples_leaf': [30,40,50],
    'min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3, 0.4],
    'max_depth': [5,6,7,8],
    'subsample': [0.6,0.7,0.8],
    'learning_rate': [0.001, 0.01, 0.09, 0.1, 0.15]
}


# In[13]:


# Random_Forest
params_grid= {'n_estimators':[2,3,4,5,6,7,8,9,10,11,12],
              "max_features":['auto'],
              'max_depth':[5,10,20,30],
              'max_leaf_nodes':[2,3,4,5],
              'min_samples_leaf':[5,10,15],
              'min_samples_split':[3,6,9,12,15,18,21]}


# In[14]:


# Adaptative Boosting
params1= {
    'n_estimators': [30,40,50],
    'learning_rate' : [0.001, 0.01, 0.09, 0.1, 0.15]}


# ## 4 -- Data_input

# In[15]:


# filepath = "POWER_Point_Daily_20130101_20221130_000d1300N_067d0890W_LST.csv"
df = pd.read_csv(r'POWER_Point_Daily_20130101_20221130_004d2530S_069d9350W_LST.csv', sep=',',
                 header=None,encoding='utf-8',skiprows=20) 
df= df.rename(columns=df.iloc[0]).loc[1:]
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) # special_character_treatment
df = df.round(decimals = 3)
cols = ["YEAR","MO","DY"]
df['date_m'] = df[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
df['date_m']=pd.to_datetime(df['date_m'])
df = df.iloc[:,3:]
first_column = df.pop('date_m')
df.insert(0, 'date_m', first_column)
print(df.shape)
df.head(2)


# In[16]:


df = df[['date_m','ALLSKY_SFC_SW_DWN','ALLSKY_KT','T2M','T2M_MAX','T2M_MIN','RH2M','PRECTOTCORR','PS','WS10M','WS10M_MAX','WS10M_MIN','WD10M']]


# In[17]:


data_float(df)
df.info()


# Input_parameters

# In[18]:


data_count(df)


# In[19]:


data_replace(df)


# In[20]:


df.info()


# #### Input & Target variables

# In[21]:


Xs = df.iloc[:,1:12]
ys = df.iloc[:,0:1]


# In[24]:


Xs.tail(5)


# train_test_split

# In[25]:


x_train = Xs.iloc[0:3617,:]
x_test = Xs.iloc[3617:3622,:] # last four months
y_train = ys.iloc[0:3617:]
y_test = ys.iloc[3617:3622,:] # last four months


# #### Data_treatment

# In[26]:


y_train = np. array(y_train)
y_rav = np.ravel(y_train, order = 'C')
y_rav_s = np.ravel(ys, order = 'C')


# ## 5 -- Feature_importance_first_check

# In[27]:


model = ExtraTreesRegressor(n_estimators=10)
model.fit(x_train, y_rav) # mostra_inicial_hiperparametros


# In[28]:


print(model.feature_importances_)
# relevant_explaining_variables


# In[29]:


(pd.Series(model.feature_importances_, index=x_train.columns)
   .nlargest(6).sort_values().plot.barh())


# ## 6 -- Ensemble _Random_Forest

# #### Random_Forest_initial

# In[30]:


rf = RandomForestRegressor(random_state = 42)
print('parameters_in_use_at_start:\n')
pprint(rf.get_params())


# #### Random_search_Random_Forest

# In[31]:


random_rf = RandomizedSearchCV(estimator = rf,cv=10,param_distributions=params_grid,n_iter=100,verbose=2,n_jobs=-1)
random_rf.fit(x_train, y_rav)


# #### Cross_validation_score_Random_Forest

# In[32]:


score_rf = cross_val_score(random_rf,Xs,y_rav_s,cv=10)
score_rf


# In[33]:


random_rf.best_params_


# In[34]:


m_rf = RandomForestRegressor(n_estimators = 4,
 min_samples_split = 9,
 min_samples_leaf = 10,
 max_leaf_nodes = 5,
 max_features = 'auto',
 max_depth = 20)


# In[35]:


yrf = m_rf.fit(x_train,y_rav)
print(yrf)


# Ensemble_application

# In[36]:


y_rf= yrf.predict(x_test)


# Ensemble_score

# In[37]:


score(y_test,y_rf)


# In[38]:


adjusted_r2(y_test,y_rf,x_test)


# In[39]:


mse_rf = mean_squared_error(y_test,y_rf)
mae_rf = mean_absolute_error(y_test,y_rf)


# In[40]:


comp_to_mean(y_train,y_test,mse_rf,mae_rf)


# ## 7 --  Ensemble_LGBoosting

# #### LGB_initial

# In[41]:


lgb = lgb.LGBMRegressor(learning_rate = 0.001,num_leaves = 65,n_estimators = 100)                       
lgb.fit(x_train, y_rav)


# #### Random_search_LGB

# In[42]:


random_search_lgb = RandomizedSearchCV(lgb,param_distributions=params_grid_g,n_iter=5,n_jobs=-1,cv=5,verbose=3)


# #### Cross_validation_score_LGB

# In[43]:


score_lgb = cross_val_score(random_search_lgb,Xs,ys,cv=10)
score_lgb


# In[44]:


random_search_lgb.fit(x_train, y_rav)


# In[45]:


print(random_search_lgb.best_estimator_)


# In[46]:


m_lgb = LGBMRegressor(booster='gbtree', boosting_type='gbdt', class_weight=None,
              colsample_bytree=0.5, gamma=0.2, importance_type='split',
              learning_rate=0.3, max_depth=5, min_child_samples=20,
              min_child_weight=0.001, min_child_weigth=1, min_split_gain=0.0,
              n_estimators=100, n_jobs=-1, num_leaves=65, objective=None,
              random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent='warn',
              subsample=1.0, subsample_for_bin=200000, subsample_freq=0)


# #### Ensemble_application

# In[47]:


m_lgb.fit(x_train,y_rav)


# In[48]:


y_lgb = m_lgb.predict(x_test)


# #### Ensemble_score

# In[49]:


score(y_test,y_lgb)


# In[50]:


adjusted_r2(y_test,y_lgb,x_test)


# In[51]:


mse_lgb= mean_squared_error(y_test,y_lgb)
mae_lgb = mean_absolute_error(y_test,y_lgb)


# In[52]:


comp_to_mean(y_train,y_test,mse_lgb,mae_lgb)


#  ## 8 -- Ensemble_XGBoost

# #### XGBoost_initial

# In[53]:


xgb = XGBRegressor()
xgb_model = XGBRegressor(learning_rate = 0.001, max_depth = 8,n_estimators = 100)
xgb_model.fit(x_train, y_train)


# #### Random_search_XGBoost

# In[54]:


random_search_xgb = RandomizedSearchCV(xgb,param_distributions=params_grid_g,n_iter=5,n_jobs=-1,cv=10,verbose=3)


# #### Cross_validation_score_XGBoost

# In[55]:


score_xgb = cross_val_score(random_search_xgb,Xs,y_rav_s,cv=10)
score_xgb


# In[56]:


random_search_xgb.fit(x_train, y_rav)


# In[57]:


print(random_search_xgb.best_estimator_)


# In[59]:


m_xgb = XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.5,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0.3, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.4, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=3, max_leaves=0, min_child_weight=1,
             min_child_weigth=3, missing= 0, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', predictor='auto', random_state=0)


# Ensemble_Xgboost_application

# In[60]:


m_xgb.fit(x_train,y_train)


# In[61]:


y_xgb = m_xgb.predict(x_test)


# #### Ensemble_score

# In[62]:


score(y_test,y_xgb)


# In[63]:


adjusted_r2(y_test,y_xgb,x_test)


# In[64]:


mse_xgb= mean_squared_error(y_test,y_xgb)
mae_xgb = mean_absolute_error(y_test,y_xgb)


# In[65]:


comp_to_mean(y_train,y_test,mse_xgb,mae_xgb)


# ## 9 -- Ensemble_Gradient_Boosting

# #### Gradient_Boosting_initial

# In[66]:


gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(x_train, y_rav)


# #### Random_search_Gradient_Boosting

# In[67]:


random_search_gbr = RandomizedSearchCV(gbr,param_distributions=params0,n_iter=5,n_jobs=-1,cv=5,verbose=3)


# #### Cross_validation_score_Gradient_Boosting

# In[68]:


score_gbr = cross_val_score(random_search_gbr,Xs,y_rav_s,cv=10)
score_gbr


# In[69]:


random_search_gbr.fit(x_train, y_rav)


# In[71]:


print(random_search_gbr.best_estimator_)


# In[72]:


m_gbr = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1, loss='ls', max_depth=5,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=50, min_samples_split=500,
                          min_weight_fraction_leaf=0, n_estimators=100,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=42, subsample=0.7, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)


# #### Ensemble_application

# In[73]:


m_gbr.fit(x_train,y_rav)


# In[74]:


y_gbr = m_gbr.predict(x_test)


# #### Ensemble_score

# In[75]:


score(y_test,y_gbr)


# In[76]:


adjusted_r2(y_test,y_gbr,x_test)


# In[77]:


mse_gbr= mean_squared_error(y_test,y_gbr)
mae_gbr = mean_absolute_error(y_test,y_gbr)


# In[78]:


comp_to_mean(y_train,y_test,mse_gbr,mae_gbr)


# ## Ensemble_Adaptive_Boosting

# #### Adaptative_Boosting_initial

# In[79]:


ada_reg = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=1), n_estimators=100,
    learning_rate=0.001)
ada_reg.fit(x_train, y_rav)


# #### Random_Search_Adaptative_Boosting

# In[80]:


random_search_ada_reg = RandomizedSearchCV(ada_reg,param_distributions=params1,n_iter=5,n_jobs=-1,cv=5,verbose=3)


# #### Cross_validation_score_Adaptative_Boosting

# In[81]:


score_ada_reg = cross_val_score(random_search_ada_reg,Xs,y_rav_s,cv=10)
score_ada_reg


# In[82]:


random_search_ada_reg.fit(x_train, y_rav)


# In[83]:


print(random_search_ada_reg.best_estimator_)


# In[84]:


m_ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(ccp_alpha=0.0,
                                                       criterion='mse',
                                                       max_depth=1,
                                                       max_features=None,
                                                       max_leaf_nodes=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0,
                                                       presort='deprecated',
                                                       random_state=None,
                                                       splitter='best'),
                  learning_rate=0.1, loss='linear', n_estimators=50,
                  random_state=None)


# In[85]:


m_ada.fit(x_train,y_rav)


# #### Ensemble_application

# In[86]:


y_ada = m_ada.predict(x_test)


# #### Ensemble_score

# In[87]:


score(y_test,y_ada)


# In[88]:


adjusted_r2(y_test,y_ada,x_test)


# In[89]:


mse_ada= mean_squared_error(y_test,y_ada)
mae_ada = mean_absolute_error(y_test,y_ada)


# In[90]:


comp_to_mean(y_train,y_test,mse_ada,mae_ada)


# In[ ]:




