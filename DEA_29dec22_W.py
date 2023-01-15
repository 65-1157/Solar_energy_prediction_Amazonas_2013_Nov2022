#!/usr/bin/env python
# coding: utf-8

# # Solar_energy_forecast_Data_Exploratory_Analysis

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
import csv
import warnings
warnings.filterwarnings('ignore')


# #### 2 -- Functions

# In[3]:


def data_dist(df):
    for a in df.columns:
        b = df[a].unique()
        sns.distplot(b)
        # plt.hist(b,11)
        plt.title(a)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()


# In[4]:


def data_float(df):
    df.set_index("date_m", inplace = True)
    for a in df.columns:
        df[a] = df[a].astype(float)
    # df.reset_index(inplace = True)


# In[5]:


def data_count(df):
    #df.set_index("date_m", inplace = True)
    for a in df.columns:
        b = -999.0
        print(a,df[a].value_counts()[b])
    #df.reset_index(inplace = True)


# In[6]:


def data_replace(df):
    #df.set_index("date_m", inplace = True)
    for a in df.columns:
        z = df[a].median()
        print(a,z)
        b = -999.0
        df[a] = df[a].replace(b,z)
    #df.reset_index(inplace = True)


# In[7]:


def data_resample_W(df):
    df = df.resample('W').mean()
    df = df.round(decimals = 3)
    return df


# In[8]:


def graph_plots(a):
    fig, axes = plt.subplots(nrows=12, ncols=1, dpi=100, figsize=(12,50))
    for i, ax in enumerate(axes.flatten()):
        data = a[a.columns[i]]
        ax.plot(data, color='blue', linewidth=2)
        # Decorations
        ax.set_title(a.columns[i])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        # ax.xaxis.set_minor_locator(mdates.MonthLocator())
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
        # ax.xaxis.set_major_formatter(
        # mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(True)
        ax.spines["top"].set_alpha(0)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize = 13)

    plt.tight_layout();


# In[9]:


def comp_graphs(a):
    # fig, axes = plt.subplots(nrows=11, ncols=1, dpi=100, figsize=(15,10))
    dfz = a.iloc[:,1:12]
    # for i, ax in enumerate(axes.flatten()):
    for i in range(0,11):
        plt.rcParams["figure.figsize"] = (10,4)
        fig, ax1 = plt.subplots()
        color = 'darkred'
        ax1.set_xlabel('date_m')
        ax1.set_ylabel('ALLSKY_SFC_SW_DWN', color=color)
        ax1.plot(a.index, a['ALLSKY_SFC_SW_DWN'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'darkblue'
        ax2.set_ylabel(dfz.columns[i], color=color)  # we already handled the x-label with ax1
        ax2.plot(a.index, dfz[dfz.columns[i]], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


# In[10]:


def box_plot(a):
    plt.rcParams["figure.figsize"] = (3, 7)
    for a in df.columns:
        b = df[a]
        sns.boxplot(b,orient = 'v')
        # plt.hist(b,11)
        plt.title(a)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Feature_W')
        plt.ylabel('Value')
        plt.show()


# #### 3 -- Data_input

# In[13]:


# filepath = "POWER_Point_Daily_20130101_20221130_000d1300N_067d0890W_LST.csv"
df = pd.read_csv(r'POWER_Point_Daily_20130101_20221130_003d1000S_060d0000W_LST.csv', sep=',',
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


# In[13]:


data_float(df)
df.info()


# In[14]:


data_count(df)


# In[15]:


data_replace(df)


# In[16]:


df.head(2)


# #### 4 -- Data_treatment

# In[17]:


df.info()


# In[18]:


df = data_resample_W(df)


# #### Data_visualization

# In[19]:


graph_plots(df)


# In[49]:


data = df.iloc[:,0:3]
plt.rcParams["figure.figsize"] = (10,4)
sns.lineplot(data = data)


# In[20]:


comp_graphs(df)


# #### Data_distribution

# In[21]:


plt.rcParams["figure.figsize"] = (5, 3)
data_dist(df)


# In[22]:


box_plot(df)


# #### Correlation_Matrix

# In[23]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(3)
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps


# In[24]:


df.describe()


# In[ ]:




