# Custom functions
import pandas as pd
import os
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
 
from utils.config_reader import config_reader

#set path
import sys
sys.path.insert(1, '../')

# Import parameters
config = config_reader('../config/config.json')

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



def plot_history_regr(history:dict=None, model_name:str=None, plot_counter:int=None):
    """Training history visualization
    
    Аргументы:
    history (keras.callbacks.History) - Training history data,
    model_name (str) - figure title. Use: model.name
    plot_counter (int) - figure id.      
    """
    mse_train =  history.history['mse'] 
    mse_val =  history.history['val_mse']  # validation sample
        
    train_loss =  history.history['loss']
    val_loss =  history.history['val_loss']

    epochs = range(len(mse_train))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

    ax[0].plot(epochs, train_loss, 'b', label='Train')
    ax[0].plot(epochs, val_loss, 'r', label='Valid')
    ax[0].set_xlabel('Epoch', size=11)
    ax[0].set_ylabel('Loss', size=11)
    ax[0].set_title('Loss')
    ax[0].legend(['train', 'val'])

    ax[1].plot(epochs, mse_train, 'b', label='Train')
    ax[1].plot(epochs, mse_val, 'r', label='Valid')
    ax[1].set_xlabel('Epoch', size=11)
    ax[1].set_ylabel('MSE value', size=11)
    ax[1].set_title(f"MSE")
    ax[1].legend(['train', 'val'])

    if plot_counter is not None:
        plt.suptitle(f"Fig.{plot_counter} - {model_name} model", y=0.05, fontsize=14)
        plt.savefig(os.path.join(config.path_figures + f'fig_{plot_counter}.png'))
    
    else: 
        plot_counter = 1
        plt.suptitle(f"Fig.{plot_counter} - {model_name} model", y=-0.1, fontsize=14)  
    plt.tight_layout();
    
    
def split_sequence(sequence:np.array, n_steps:int)->np.array:
	""" Split a univariate sequence into samples

	Args:
		sequence (np.array): time series (1 feature)
		n_steps (int): lag

	Returns:
		np.array: _description_
	"""    
	X, y = list(), list()
 
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
  
	return np.array(X), np.array(y)

def callbacks(model_name:str, lr:float=config.model.learning_rate, save_model:bool=True):
    """Model training setup function

    Args:
        min_lr (_float_): lower boundary of the learning rate to stop training
        monitor (str) - metric name 
        mode (str)- modes {"auto", "min", "max"}. Max - stop training if the metric doesn't improve
        reduce_patience (_int_): number of epochs to evaluate the learning rate improvement
        stop_patience (_int_):  number of epochs before training will be ended
        path_models (_str_): path to save the model
        save_best_only (bool): If True then saves the model with the best metric.
    """      
    
    # Save the best model
    checkpoint = ModelCheckpoint(
        os.path.join(config.path_models, model_name + '.hdf5'), 
        monitor=config.monitor, 
        verbose=config.verbose, 
        mode=config.mode, 
        save_best_only=save_model
    )

    # stop training if the metric doesn't improve
    earlystop = EarlyStopping(
        monitor=config.monitor, 
        mode=config.mode, 
        patience=config.stop_patience, 
        restore_best_weights=config.restore_best_weights
    )

    # reduce the learning rate if the metric doesn't improve
    reduce_lr = ReduceLROnPlateau(
        monitor=config.monitor, 
        mode=config.mode,  
        factor=config.factor, 
        patience=config.reduce_patience,  # might be 10
        verbose=config.verbose, 
        min_lr=lr/1000
    )
    
    return [checkpoint, earlystop, reduce_lr]