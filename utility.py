# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import process as proc
import numpy as np
   
def symbol_to_path(name, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}".format(str(name)))
    #return os.path.join(base_dir, "{}.csv".format(str(name)))

def plot_selected(df, columns, start, end, title="data"):
    """Plot the desired columns over index values in the given range."""
    plot_data(df.ix[start:end, columns], title)

def plot_data(df, title="", xlable="date", ylable="price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlable)
    ax.set_ylabel(ylable)
    plt.show()

def plot_histogram(df, title="", xlable="date", ylable="price"): 
    for idx in df:    
        df[idx].hist(bins=30, label=idx)
        
    #plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.legend(loc='upper right')
    plt.show()

def plot_stock_CAPM(stocks, target, title):
    daily_returns = proc.get_daily_returns(stocks)
    daily_returns.plot(kind='scatter', x='SPY', y=target)
    beta, alpha = np.polyfit(daily_returns['SPY'], daily_returns[target], 1)
    #Plot the fit line
    plt.plot(daily_returns['SPY'], beta*daily_returns['SPY']+alpha, '-', color = 'r')
    plt.title(title)
    plt.show()

def save_data(name, df):
    df.to_csv(symbol_to_path(name), sep=',')
    return




