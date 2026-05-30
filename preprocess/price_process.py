from torch.utils.data.dataset import Dataset
import torch.utils.data.dataloader as DataLoader
import pandas as pd
import numpy as np
import torch 
from globals import LookBack, indicators, WEEKDAY, OCLH, OCLHV, kdj
from functools import partial
from stockstats import StockDataFrame
from transformers import BertModel, BertTokenizer
import re, json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from utils import validate

################ add next-day directional label ################
def attach_label(df):
    '''
    Directional label: y_t = 1 if x_{t+1}^Close > x_t^Close (bullish),
                         y_t = 0 if x_{t+1}^Close < x_t^Close (bearish).
    Tie cases carry forward the previous label.
    '''
    df['next_close'] = df['close'].shift(-1)
    df['label_raw'] = np.where(df['next_close'] > df['close'], 1,
                               np.where(df['next_close'] < df['close'], 0, np.nan))
    # carry-forward for tie cases (y_t = y_{t-1})
    df['label_raw'] = df['label_raw'].fillna(method='ffill')
    # if the very first row is still NaN, default to 0
    df['label_raw'] = df['label_raw'].fillna(0)
    df['label'] = df['label_raw'].astype(int)
    df = df.drop(columns=['next_close', 'label_raw'])



########### add indicators ###########

def add_indicators(df):
    sdf = StockDataFrame(df.copy()) 
    # 可以改为sdf[WEEKDAY], global部分要改为WEEKDAY='weekday'
    df['weekday'] = pd.to_datetime(sdf.index).dayofweek
    # kdj 要单独计算，stockstat调用不出来
    df[indicators] = sdf[indicators]
    df['kdjk'], df['kdjd'],df['kdjj'] = calculate_kdj(df)
    df = df.dropna()
    return df

def calculate_kdj(data, period=9, k_period=3, d_period=3):
    low_min = data['low'].rolling(window=period).min()
    high_max = data['high'].rolling(window=period).max()
    
    rsv = (data['close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(0) 
    
    k = rsv.ewm(com=(k_period-1), adjust=False).mean()
    d = k.ewm(com=(d_period-1), adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))


def process_price(price_df, selected_feature=indicators+kdj, data_type='carbon', method='dualcl',
                  lookahead=5, lookback=20):
    # 计算星期几
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df['weekday'] = price_df['date'].dt.weekday
    weekday_df = pd.get_dummies(price_df['weekday'], prefix='weekday')
    price_df = pd.concat([price_df.drop(columns=['weekday']), weekday_df], axis=1)
    weekcol = ['weekday_'+str(i) for i in range(5)]
    
    price_df = price_df.sort_values(by='date', ascending=True).set_index('date')
    price_df = add_indicators(price_df)
    attach_label(price_df)  # next-day directional label
    if method == 'dtml':
        return price_df.dropna()[OCLHV+ selected_feature + weekcol+ ['label']]
    else:
        return price_df.dropna()[OCLHV+ selected_feature+ ['label']]
        
        
def shift_datetime(date, shift_from='%Y%m%d', shift_to='%Y%m%d'):
    date = pd.to_datetime(date, format=shift_from)
    date = date.dt.strftime(shift_to)
    return date

def process_carbon(price_df):
    price_df = price_df.drop(columns =['name'])
    price_df['low'] = pd.to_numeric(price_df['low'], errors='coerce')
    price_df['high'] = pd.to_numeric(price_df['high'], errors='coerce')  
    price_df['date'] = shift_datetime(price_df['date'],shift_from='%Y%m%d', shift_to='%Y-%m-%d')
    price_df = price_df.sort_values(by='date', ascending=True).set_index('date')
    return price_df