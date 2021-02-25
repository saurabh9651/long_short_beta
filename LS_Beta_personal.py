# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:48:43 2017

@author: patel_saurabh_j
"""
#%%
from IPython import get_ipython
get_ipython().magic('reset -sf')


import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import statsmodels.formula.api as smf
import statsmodels.api as sm

import pandas_datareader.data
import fix_yahoo_finance as web
from pandas_datareader.famafrench import get_available_datasets


path = 'C:\\fakepath\\OneDrive\\001_Personal_Porfolio_Python_code\\'

store_data = path + "MyPorfolio_Stock_Data_MSR_MIN.h5"
#%% prepare prices and factors
#%%
#
#t = pd.read_csv(path + 'NSE_tickerlist.csv')
#tickers = t['tickers'].tolist()
#tickers.extend(['^NSEI'])

t = ('HMVL', 'WIPRO', 'SHEMAROO', 'FINCABLES', 'BEL', 'LT', 'EMAMILTD', 'GLENMARK', 'TATAMOTORS', 'ONGC', 'GRASIM', 'TCS', 'CAPF', 'TECHM', 'HDFCBANK', 'IOC', 'INDUSINDBK', 'MARUTI', 'PIIND', 'TVSMOTOR', 'ZEEL', '^NSEI')
exch = '.NS' #For NSE quotes
#exch = '.BO' #For BSE quotes
tickers = list()
for i in t:
    if i == '^NSEI':
        tickers.append(i)
    else:
        j = i + exch
        tickers.append(j)
    
#%%
start       = '2008-1-1'
end         = dt.datetime.now()
#
stk_data    = web.download(tickers, start, end)
stk_data.rename(items={'Adj Close': 'AdjClose'},inplace=True)
stk_data.rename(minor_axis={'^NSEI': 'NIFTY50'},inplace=True)
stk_adj         = stk_data.AdjClose
stk_adjcl       = stk_adj.fillna(method = 'ffill')
stk_adjcl_prc   = stk_adjcl.fillna(method = 'bfill')
stk_adjcl_prc   = stk_adjcl_prc.dropna(axis=1)
bchmk_adjcl_prc = stk_adjcl_prc['NIFTY50']
del stk_adjcl_prc['NIFTY50']

t_2 = [s.strip('.NS') for s in stk_adjcl_prc.columns]
t_3 = [s.strip('-EQ') for s in t_2]
t_4 = [s.strip('-P2') for s in t_3]
t_5 = [s.strip('-MF') for s in t_4]
t_6 = [w.replace('J&KBANK', 'JandKBANK') for w in t_5]

stk_adjcl_prc.columns = t_6
stk_ret         = stk_adjcl_prc.pct_change()

ffm_1           = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start)
ffm_data        = pd.DataFrame(ffm_1[0]/100)
ffm_data.columns = ['mktrf','smb','hml','rf']

#
store           = pd.HDFStore(store_data)
store['stk_data_1']      = stk_data
store['ffm_data_1']      = ffm_data
store.close()



#stk_data = pd.read_hdf(path + "MyPorfolio_Stock_Data_MSR_MIN.h5", 'stk_data')
#stk_adjcl_prc = stk_data.AdjClose
#stk_adjcl_prc = stk_data.AdjClose
#bchmk_adjcl_prc = stk_adjcl_prc['NIFTY50']
#del stk_adjcl_prc['NIFTY50']
#stk_ret = stk_adjcl_prc.pct_change()

#%%
axis_dates          = stk_data.major_axis
alldates            = pd.DataFrame(axis_dates,index=axis_dates)
alleom              = alldates.groupby([alldates.index.year,alldates.index.month]).last()
alleom.index        = alleom.Date
axis_eom            = alleom.index
axis_id             = stk_adjcl_prc.columns
#%%
def Ptf_Sharpe_Ratio(w, ret):
    vcv = ret.cov()
    mu = ret.mean()
    num = w.dot(mu.T)
    den = (w.dot(vcv).dot(w.T))**(0.5)
    sharpe_ratio = num/den
    return sharpe_ratio*-1

def Ptf_Variance(w, ret):
    vcv = ret.cov()
    var = w.dot(vcv).dot(w.T)
    return var

#n = len(stk_adjcl_prc.columns)    
#ret = stk_ret
#w = np.ones((1,n))/n   
#vcv = ret.cov()
#mu = ret.mean()
#num = w.dot(mu.T)
#den = (w.dot(vcv).dot(w.T))**(0.5)
#sharpe_ratio = num/den
#%% estimate betas 

stk_ret_ff = stk_ret.join(ffm_data.mktrf,how='inner')

betas = pd.DataFrame([],index=axis_eom,columns=axis_id)
weights_eom     = pd.DataFrame([],index=axis_eom,columns=axis_id)
n = len(stk_adjcl_prc.columns)

for t in axis_eom[13::]:
    dfret1 = stk_ret_ff.loc[t-pd.DateOffset(months = 12):t,:]
    for id in axis_id:
        formula = id + '~' + 'mktrf'
        res = smf.ols(formula = formula,data = dfret1).fit()
        betas.loc[t,id] = res.params[1]
    b = betas.loc[t,:]
    m = b.median()
    weights_eom.loc[t,axis_id[b<m]] = 1/b[b<m].mean()/(b<m).sum()
    weights_eom.loc[t,axis_id[b>m]] = -1/b[b>m].mean()/(b>m).sum()
    

#betas.to_csv(fn_bb)
#test4 = pd.read_csv(fn_bb)
#test4.set_index('Date',inplace=True)
#
#store        = pd.HDFStore(fn_bbh5)
#store['betas']  = betas
#store.close()
#
#store       = pd.HDFStore(fn_wh5)
#store['w']  = w
#store.close()
#
#test3 = pd.read_hdf(fn_bbh5,'betas')
#test5 = pd.read_hdf(fn_wh5,'w')

#%% backtest 
weights_daily = pd.DataFrame(weights_eom, index = axis_dates, columns = axis_id)
weights_daily = weights_daily.fillna(method = 'ffill')

eq_w = 1/n
weights_daily_eq = pd.DataFrame(eq_w, index = axis_dates, columns = axis_id)
weights_eom_eqw = pd.DataFrame(weights_daily_eq, index = axis_eom, columns = axis_id)
#%%
ptf_value_lsbeta = pd.DataFrame([], index = axis_dates, columns = ['Ptf_Value_LSBeta'])
ptf_value_eqw = pd.DataFrame([], index = axis_dates, columns = ['Ptf_Value_EQW'])
for t in axis_dates:
    ptf_value_lsbeta.loc[t] = weights_daily.loc[t].dot(stk_adjcl_prc.loc[t].T)
    ptf_value_eqw.loc[t] = weights_daily_eq.loc[t].dot(stk_adjcl_prc.loc[t].T)
#%%
tt = '2017-01-02'

hundred_base_lsbeta = pd.DataFrame([], index = axis_dates, columns = ['LSBeta'])
hundred_base_lsbeta = hundred_base_lsbeta[tt:]

hundred_base_eqw = pd.DataFrame([], index = axis_dates, columns = ['EQW'])
hundred_base_eqw = hundred_base_eqw[tt:]

hundred_base_bchmk = pd.DataFrame([], index = axis_dates, columns = ['BCHMK'])
hundred_base_bchmk = hundred_base_bchmk[tt:]

for t in hundred_base_lsbeta.index:
    hundred_base_lsbeta.loc[t] = np.array(ptf_value_lsbeta.loc[t]*100)/np.array(ptf_value_lsbeta.loc[tt])
    hundred_base_eqw.loc[t] = np.array(ptf_value_eqw.loc[t]*100)/np.array(ptf_value_eqw.loc[tt])
    hundred_base_bchmk.loc[t] = np.array(bchmk_adjcl_prc.loc[t]*100)/np.array(bchmk_adjcl_prc.loc[tt])
        
all_val = pd.concat([hundred_base_lsbeta, hundred_base_eqw, hundred_base_bchmk], axis=1, join='inner')
#all_val.plot()
#%%

oos = tt
ptf_ret_lsbeta = ptf_value_lsbeta.loc[oos::].pct_change()
ptf_ret_eqw = ptf_value_eqw.loc[oos::].pct_change()
ptf_ret_bchmk = hundred_base_bchmk.loc[oos::].pct_change()

sharpe_ratio_lsbeta = (ptf_ret_lsbeta.mean()*252) / (ptf_ret_lsbeta.std()*((252)**0.5))
sharpe_ratio_eqw = (ptf_ret_eqw.mean()*252) / (ptf_ret_eqw.std()*((252)**0.5))
sharpe_ratio_bchmk = (ptf_ret_bchmk.mean()*252) / (ptf_ret_bchmk.std()*((252)**0.5))
#%%

all_val.plot()
print ('Long Short Beta Strategy Sharpe Ratio')
print (sharpe_ratio_lsbeta)
print ('Equal Weighted Optimization OOS Sharpe Ratio')
print (sharpe_ratio_eqw)
#%%Last Day weight and Units to be purchased
notional = 54000

today = end

todays_weight_units_lsbeta = (weights_daily.iloc[-1] * notional) / (stk_adjcl_prc.iloc[-1])
todays_weight_units_eqw = (weights_daily_eq.iloc[-1] * notional) / (stk_adjcl_prc.iloc[-1])

print ('Long Short Beta Strategy Weights')
print (todays_weight_units_lsbeta.astype(dtype = float).round(decimals = 0))
print (todays_weight_units_lsbeta.astype(dtype = float))

print ((todays_weight_units_lsbeta.astype(dtype = float).round(decimals = 0) * stk_adjcl_prc.iloc[-1]).sum(0))
print ((todays_weight_units_lsbeta * stk_adjcl_prc.iloc[-1]).sum(0))