# Custom functions
import pandas as pd
import os
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt

 

def Dickey_Fuller_test(data:pd.Series): #
    """Augmented Dickey-Fuller unit root test on stationarity of series

    Args:
        data (pd.Series): data 
       
    Returns:
        data (pd.Series): differentiated data
        diff_order (int): difference order
    """ 
   
    diff_order = 0
                
    while True:
        
        test1 = adfuller(data)
        
        if test1[0] > test1[4]['5%']: 
            
            data = data.diff().dropna()
            diff_order += 1
            print('Attempt {}: non-stationary series'.format(diff_order))
            
        else:
            
            print("ADF Statistic: {:.2f}".format(test1[0]))
            print("p-value: {:.2g}".format(test1[1]))
            print("Critical Values:")

            for key, value in test1[4].items():
                print(f"\t{key}: {value:.3f}")
                
            print('Stationary series\nDiff order = {}'.format(diff_order))
            break
        
    return data, diff_order 

def optim_param(data:pd.DataFrame, field:np.array)->pd.DataFrame:
    """Find optimized parameters of the ARIMA model 

    Args:
        field (np.array): range of parameters

    Returns:
        params (pd.DataFrame): The dataframe of estimated model parmeters sorted by AIC, which is the model quality.
    """    
    params = {
    'p':[],# last significant lag taken from the partial autocorrelation graph
    'd':[],# difference order
    'q':[],# last significant lag taken from the  autocorrelation graph
    'aic':[] # aic score
    }
    #field = np.array([])
    
    
    for p in field:
        for d in field:
            for q in field:
                arima_model = ARIMA(data, order=(p, d, q))
                arima_model_fit = arima_model.fit()

                params['p'].append(p)
                params['d'].append(d)
                params['q'].append(q)
                params['aic'].append(arima_model_fit.aic.round(2))
    
    return pd.DataFrame(params).sort_values(by='aic', ascending=False)


def plot_acf_pacf(series: pd.Series, lags:int, fig_id:int=1, zero:bool=False)->None:
    """Plots two graph: autocorrelation and partial autocorrelation

    Args:
        series (pd.Series): data
        lags (int): number of lags to display on the plot
        fig_id (int): Number of the figure
        zero (bool, optional): Flag indicating whether to include the 0-lag autocorrelation. Default is False.
    """     
     
    fig, axes = plt.subplots(1, 2, figsize=(9,4))

    sgt.plot_acf(series, ax=axes[0], lags=lags, zero=zero)
    sgt.plot_pacf(series, ax=axes[1], lags=lags, zero=zero, method="ywm")
    #major_ticks = np.arange(lags)
    #plt.xticks(major_ticks)
    plt.suptitle("Fig.{} - Autocorrelation and partial autocorrelation".format(fig_id), y=-0.05)
    plt.show()