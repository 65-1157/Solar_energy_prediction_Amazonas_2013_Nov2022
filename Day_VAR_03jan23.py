#!/usr/bin/env python
# coding: utf-8

# # Solar_energy_forecast_Vector_Autoregression_Amazon
# #### Reference1 = https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# #### Reference2 = https://towardsdatascience.com/a-quick-introduction-on-granger-causality-testing-for-time-series-analysis-7113dc9420d2
# #### Reference3 = https://analyticsindiamag.com/hands-on-tutorial-on-vector-autoregressionvar-for-time-series-modeling/

# #### 1 -- Load_Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf
import re
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')


# #### 2 -- Functions

# In[3]:


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the output, columns are the input or predictors. 
    The values in the table are the P-Values. 
    If the values are smaller than 0.05, then the input influences the output.
    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


# In[4]:


def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary.
    """
    out = coint_johansen(df,-1,12)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)
    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)


# In[5]:


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


# In[6]:


def invert_transformation(df_train, df1_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df1_fc = df1_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df1_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df1_fc[str(col)+'_2d'].cumsum()
            
        # Roll back 1st Diff
        df1_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df1_fc[str(col)+'_1d'].cumsum()
        
    return df1_fc


# In[7]:


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})


# In[8]:


def data_float(df):
    df.set_index("date_m", inplace = True)
    for a in df.columns:
        df[a] = df[a].astype(float)
    # df.reset_index(inplace = True)


# In[9]:


def data_count(df):
    #df.set_index("date_m", inplace = True)
    for a in df.columns:
        b = -999.0
        print(a,df[a].value_counts()[b])
    #df.reset_index(inplace = True)


# In[10]:


def data_replace(df):
    #df.set_index("date_m", inplace = True)
    for a in df.columns:
        z = df[a].median()
        print(a,z)
        b = -999.0
        df[a] = df[a].replace(b,z)
    #df.reset_index(inplace = True)


# In[11]:


def data_resample_W(df):
    df = df.resample('W').mean()
    df = df.round(decimals = 3)
    return df


# #### 3 -- Data_input

# In[12]:


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


# #### 4 -- Data_treatment

# In[13]:


data_float(df)
df.info()


# In[14]:


data_count(df)


# In[15]:


data_replace(df)


# #### Check for Stationarity and Make the Time Series Stationary

# In[16]:


# ADF Test on each parameter
for name, column in df.iteritems():
    adfuller_test(column, name=column.name)
    print('********************************************')


# #### Train_Test_Split

# In[60]:


nobs = 1
df_train, df_test = df[0:-nobs], df[-nobs:]
# Check size
print(df_train.shape,df_test.shape) 


# #### Selection of the Order (P) of VAR model

# ##### Method_1

# In[61]:


model = VAR(df_train)
for i in range(1,15):
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')


# ##### Method_2 

# In[65]:


x = model.select_order(maxlags=4)
x.summary()


# #### Train the VAR Model of Selected Order(p)

# In[66]:


model_fitted = model.fit(4)
model_fitted.summary()


# #### Durbin_Watson autocorrelation check in residuals (Errors) 

# In[67]:


out = durbin_watson(model_fitted.resid)
for col, val in zip(df_train.columns, out):
    print(col, ':', round(val, 2))


# #### Granger’s Causality Test

# In[68]:


maxlag= 4
test = 'ssr_chi2test'
grangers_causation_matrix(df_train, variables = df.columns) 


# #### Cointegration Test

# In[69]:


cointegration_test(df_train, alpha=0.05)


# #### How to Forecast VAR model using statsmodels

# In[70]:


# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  

# Input data for forecasting
forecast_input = df_train.values[-lag_order:]
forecast_input


# In[71]:


# Fprecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df1_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_1d')
df1_forecast


# In[72]:


df_test.head(5)


# #### Plot of Forecast vs Actuals

# In[ ]:


fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(15,25))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df1_forecast[col+'_1d'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=10)

plt.tight_layout();


# #### Evaluate the Forecasts

# In[ ]:


lista1 = ['ALLSKY_SFC_SW_DWN','ALLSKY_KT','T2M','T2M_MAX','T2M_MIN','RH2M','PRECTOTCORR','PS','WS10M','WS10M_MAX','WS10M_MIN','WD10M']
lista2 = ['ALLSKY_SFC_SW_DWN_1d','ALLSKY_KT_1d','T2M_1d','T2M_MAX_1d','T2M_MIN_1d','RH2M_1d','PRECTOTCORR_1d','PS_1d','WS10M_1d','WS10M_MAX_1d','WS10M_MIN_1d','WD10M_1d']


# In[59]:


for a,b in zip(lista1,lista2):
    print('Forecast Accuracy of:'+ a)
    accuracy_prod = forecast_accuracy(df1_forecast[b].values, df_test[a])
    for k, v in accuracy_prod.items():
        print(k, ': ', round(v,4))
        print('======================')


# In[ ]:




