# -*- coding: utf-8 -*-
import quandl
import pandas as pd
import numpy as np
import scipy.stats as scs
import utility as util

def get_mkt_data(date_index, *args):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    # Define a date range
    df = pd.DataFrame(index=date_index)
    
    for it in args:
        for feature in it:
            if feature[-5:] == 'Price':
                df_temp = it[feature]
                df = df.join(df_temp)
            elif feature[-6:] == 'Volume':
                df_temp = it[feature]
                df = df.join(df_temp)
            
    return df

def read_mkt_data(name):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.read_csv(util.symbol_to_path(name), index_col='Unnamed: 0', parse_dates=True, na_values=['nan'])
    
    return df

def normalize_std(df):
    """Normalize stock prices using the first row of the dataframe"""    
    #df = df/df.ix[0,:]
    norm_df = (df - df.mean())/df.std()
    return norm_df

def normalize_firstrow(df):
    """Normalize stock prices using the first row of the dataframe"""    
    return df/df.ix[0,:]

def get_stocks(name, date_index, date_range, start, end):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    list_df = pd.read_csv(util.symbol_to_path(name), parse_dates=True)
    list_df = list_df[['free_code', 'name', 'ticker']]
    
    df = pd.DataFrame(index=date_index)
    
    for i in range(start, end):
       name = list_df.iloc[i]['name']  
       free_code = list_df.iloc[i]['free_code']
       ticker = list_df.iloc[i]['ticker']  
       df_temp = quandl.get(free_code, start_date=date_range[0], end_date=date_range[1])
       df_temp = df_temp.rename(columns = {'Adj. Close':ticker})
       df_temp = df_temp[ticker]
       df = df.join(df_temp)
       print "{} Download & Parse complete".format(name)
    
    return df

def get_total_stocks(name, date_index, date_range):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    list_df = pd.read_csv(util.symbol_to_path(name), parse_dates=True)
    list_df = list_df[['free_code', 'name', 'ticker']]
    
    df = pd.DataFrame(index=date_index)
    
    for ind, val in list_df.iterrows():
        name = val['name']  
        free_code = val['free_code']
        ticker = val['ticker']
        df_temp = quandl.get(free_code, start_date=date_range[0], end_date=date_range[1])
        df_temp = df_temp.rename(columns = {'Adj. Close':ticker})
        df_temp = df_temp[ticker]
        df = df.join(df_temp)
        print "{}. {} Download & Parse complete".format(ind, name)
    
    return df

def read_stock_data(name, verbose):
    df = pd.read_csv(util.symbol_to_path(name), index_col='Unnamed: 0', parse_dates=True)
    
    if verbose == True:
        print "List of companies dropped due to insufficient data"
    for column in df:
        if np.isnan(df[column].loc['2010-01-04']) or np.isnan(df[column].loc['2016-12-30']):
            df.drop(column, axis = 1, inplace = True)
            if verbose == True:
                print "{}".format(column)
    
    return df    
    
def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window, center=False).mean()

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window, center=False).std()

def get_daily_returns(df):
    daily_returns = df.copy()
    #Compute daily returns for row 1 onwards
    #when working on two arrays, numpy tends to match index in doing it, so we use values
    daily_returns[1:] = (df[1:] / df[:-1].values)
    #Set daily returns for row 0 to 0
    daily_returns.ix[0] = 1
    #daily_returns.ix[0, :] = 0 
    return daily_returns

def get_cumulative_return(df, days):
    cumulative_returns = df.copy()
    #Compute daily returns for row 1 onwards
    #when working on two arrays, numpy tends to match index in doing it, so we use values
    cumulative_returns[days:] = (df[days:] / df[:-days].values)
    #Set daily returns for row 0 to 0
    cumulative_returns.ix[:days] = 1
    return cumulative_returns    

def get_future_return(df, days):
    future_returns = df.copy()
    #Compute daily returns for row 1 onwards
    #when working on two arrays, numpy tends to match index in doing it, so we use values
    future_returns[:-days] = (df[days:] / df[:-days].values)
    #Set daily returns for row 0 to 0
    future_returns.ix[days:] = 1 
    return future_returns  

def fill_missing_values(df_data):
    # Foward first, then Backward
    df_data.fillna(method="ffill",inplace="TRUE")
    df_data.fillna(method="bfill",inplace="TRUE")
    return df_data

def price_to_return(df, target, replace, days):
    data = df.copy()
    for column in df:
        if column[-len(target):] == target:
            df_temp = data[column]
            df_temp = get_cumulative_return(df_temp, days)
            df_temp.name = df_temp.name.replace(target, replace)
            #if column[:3] == 'SPY':
            #    df_weekly = get_rolling_mean(df_daily, 5)
            #    df_weekly.name = df_weekly.name.replace('Return', 'Return(W)')
            data.drop(column, axis = 1, inplace = True)     
            data = data.join(df_temp)
            #if column[:3] == 'SPY':
            #    data = data.join(df_weekly)
      
    return data

def extract_columns(df,  *args):
    data = df.copy()
    for column in df:
        find = False
        for it in args:
            if it in column:
                find = True
        if find == False:
            data.drop(column, axis = 1, inplace = True)     
            
    return data

def print_statistics(array):    
    sta = scs.describe(array)
    print "%14s %15s" % ('statistic', 'value')
    print 30 * "-"
    print "%14s %15.5f" % ('size', sta[0])
    print "%14s %15.5f" % ('min', sta[1][0])
    print "%14s %15.5f" % ('max', sta[1][1])
    print "%14s %15.5f" % ('mean', sta[2])
    print "%14s %15.5f" % ('std', np.sqrt(sta[3]))
    print "%14s %15.5f" % ('skew', sta[4])
    print "%14s %15.5f" % ('kurtosis', sta[5])

def target_cluster_three(df, low_threshold, high_threshold):
    label = df.copy()
    for idx, val in df.iteritems():
        if val > high_threshold:
            label.values[idx] = 1
        elif val < low_threshold:
            label.values[idx] = -1
        else:
            label.values[idx] = 0
    
    return label

def target_cluster_two(df, threshold):
    label = df.copy()
    for idx, val in df.iteritems():
        if val >= threshold:
            label.values[idx] = 1
        elif val < threshold:
            label.values[idx] = 0
        
    return label
        
        
    