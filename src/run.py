from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
#from sklearn.model_selection import ShuffleSplit
import pandas as pd
import process as proc
import numpy as np
import scipy.optimize as sco

def r2_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)   
    # Return the score
    return score

def accuracy_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = accuracy_score(y_true, y_predict)   
    # Return the score
    return score

def fit_regression(reg, X, y, params):
    """ Performs grid search over passed parameter for a 
        passed regressor trained on the input data [X, y]. """
    #Cross Validation Scheme
    tscv = TimeSeriesSplit(n_splits=10)
    #Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(r2_metric)
    #Create the grid search object
    grid = GridSearchCV(reg, params, scoring = scoring_fnc, cv = tscv)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    # Return the optimal model after fitting the data
    return grid.best_estimator_

def fit_classifier(clf, X, y, params):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    #tscv = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    tscv = TimeSeriesSplit(n_splits=10)
    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(accuracy_metric)
    # TODO: Create the grid search object
    grid = GridSearchCV(clf, params, scoring = scoring_fnc, cv = tscv)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def get_r2_score(reg, features, target):
    ''' Makes predictions using a fit classifier based on accuracy score. '''
    y_pred = reg.predict(features)
    # Print and return results
    return r2_score(target.values, y_pred)

def get_accuracy_score(clf, features, target):
    ''' Makes predictions using a fit classifier based on accuracy score. '''
    y_pred = clf.predict(features)
    # Print and return results
    return accuracy_score(target.values, y_pred)

def get_stocks_CAPM(stocks):   
    df = pd.DataFrame(columns=['beta','alpha', 'corr'])
    daily_return = proc.get_daily_returns(stocks)
    corr = daily_return.corr(method='pearson')

    for col in daily_return:
        beta, alpha = np.polyfit(daily_return['SPY'], daily_return[col], 1)
        df.ix[col] = [beta, alpha, corr['SPY'][col]]

    return df

#res = capm.sort_values(['beta'], ascending=[0]) 
#res = res[(res['corr'] > res['corr'].quantile(0.75))]
#print res

def get_high_beta_stocks(stocks, date, rollback, num, threshold):
    end = stocks.index.get_loc(date)
    start = end - rollback
    daily_return = proc.get_daily_returns(stocks.iloc[start:end])  
    corr = daily_return.corr(method='pearson')
    
    capm = pd.DataFrame(columns=['beta','alpha', 'corr'])
    for col in daily_return:
        beta, alpha = np.polyfit(daily_return['SPY'], daily_return[col], 1)
        capm.ix[col] = [beta, alpha, corr['SPY'][col]]
    
    capm = capm.sort_values(['beta'], ascending=[0]) 
    capm = capm[(capm['corr'] > capm['corr'].quantile(threshold))]
    stock_tickers = capm[:num].index.get_values()  
    return stock_tickers
    
def get_low_beta_stocks(stocks, date, rollback, num, threshold):
    end = stocks.index.get_loc(date)
    start = end - rollback
    daily_return = proc.get_daily_returns(stocks.iloc[start:end])  
    corr = daily_return.corr(method='pearson')
    
    capm = pd.DataFrame(columns=['beta','alpha', 'corr'])
    for col in daily_return:
        beta, alpha = np.polyfit(daily_return['SPY'], daily_return[col], 1)
        capm.ix[col] = [beta, alpha, corr['SPY'][col]]
    
    capm = capm.sort_values(['beta'], ascending=[1]) 
    capm = capm[(capm['corr'] > capm['corr'].quantile(threshold))]
    stock_tickers = capm[:num].index.get_values()  
    return stock_tickers
   
ret = pd.DataFrame()
def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(ret.mean() * weights)
    pvol = np.sqrt(np.dot(weights.T, np.dot(ret.cov(), weights)))
    return np.array([pret, pvol, pret/pvol])

def min_func_sharpe(weights):
    return -statistics(weights)[2]

def min_func_return(weights):
    return -statistics(weights)[0]

def max_func_risk(weights):
    return statistics(weights)[1]

def optimize_weight(stocks, symbols, date, rollback, method = 'sharpe'):
    noa = len(symbols)
    portfolio = stocks[symbols]
    end = stocks.index.get_loc(date)
    start = end - rollback
    global ret
    ret = proc.get_daily_returns(portfolio.iloc[start:end])  
      
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1})
    bnds = tuple((0,1) for x in range(noa))
        
    if method == 'sharpe':
        opts = sco.minimize(min_func_sharpe, noa * [1./noa,], method='SLSQP', bounds=bnds, constraints=cons)
    elif method == 'return':
        opts = sco.minimize(min_func_return, noa * [1./noa,], method='SLSQP', bounds=bnds, constraints=cons)
    elif method == 'risk':
        opts = sco.minimize(max_func_risk, noa * [1./noa,], method='SLSQP', bounds=bnds, constraints=cons)
    #print statistics(opts['x']).round(3)
    return opts['x'].round(3)
    

def simulation(reg, X, y, stocks, date_range, rollback, n_company, corr, method = 'sharpe'):
    stock_return = proc.get_daily_returns(stocks)
    date_index = stocks.ix[date_range[0]:date_range[1]].index.get_values()
    portfolio_df = pd.DataFrame(index=date_index, columns=['value', 'return','market'])
    portfolio = 1
    portfolio_df.ix[date_index[0]] = [portfolio, 1, 1]
    # First Day
    pred = reg.predict(np.array(X.ix[date_index[0]]).reshape((1, -1)))
    if pred >= 1.0:
        market = 'bull'
        symbols = get_high_beta_stocks(stocks, date_index[0], rollback, n_company, corr)
        weights = optimize_weight(stocks, symbols, date_index[0], rollback, method)
    else:
        market = 'bear'
        symbols = get_low_beta_stocks(stocks, date_index[0], rollback, n_company, corr)
        weights = optimize_weight(stocks, symbols, date_index[0], rollback, method)
    #Iterate
    date_index = date_index[1:]
   
    for today in date_index:
       #Calculate Portfolio Return
       portfolio = sum(stock_return[symbols].ix[today].tolist()*(weights*portfolio))
       if market == 'bull':
           portfolio_df.ix[today] = [portfolio, sum(stock_return[symbols].ix[today].tolist()*weights), 1]
       elif market == 'bear':
           portfolio_df.ix[today] = [portfolio, sum(stock_return[symbols].ix[today].tolist()*weights), -1]
       #Calculate Market
       pred = reg.predict(np.array(X.ix[today]).reshape((1, -1)))
       if pred >= 1.0 and market == 'bear':
           market = 'bull'
           symbols = get_high_beta_stocks(stocks, today, rollback, n_company, corr)
           weights = optimize_weight(stocks, symbols, today, rollback, method)
           print "[{}]Market status change: bear -> bull, current portfolio value: {}".format(today, portfolio)
       elif pred < 1.0 and market == 'bull':
           market = 'bear'
           symbols = get_low_beta_stocks(stocks, today, rollback, n_company, corr)
           weights = optimize_weight(stocks, symbols, today, rollback, method)
           print "[{}]Market status change: bull -> bear, current portfolio value: {}".format(today, portfolio)
       
    return portfolio_df