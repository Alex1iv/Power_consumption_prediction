import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from keras.models import Sequential
from keras.layers import LSTM, Dense

#set path
import os
import sys
sys.path.insert(1, '../')
from functions import split_sequence, callbacks
from config_reader import config_reader

# Import parameters
config = config_reader('../config/config.json')
random_state = config.random_seed
path_figures = config.path_figures
path_models = config.path_models

tf.random.set_seed(random_state)

df = pd.read_csv("../data/energy_demand_OPSD.zip", sep=",",index_col="Date", parse_dates=['Date']) 

# Selection
X = df[['Consumption']]

# Scaling
standardScaler = StandardScaler()

X_scaled = pd.DataFrame(
    standardScaler.fit_transform(X), 
    columns=X.columns,
    index=X.index
)

# choose a number of time steps (every 7 days)
n_steps = 7

# split into samples-------------------------------
X_splitted, y_splitted = split_sequence(X_scaled.values, n_steps)
#print(X_splitted.shape,  y_splitted.shape)

X_train, X_test = X_splitted[:len(X)-100], X_splitted[len(X)-100:] 
y_train, y_test = y_splitted[:len(X)-100], y_splitted[len(X)-100:]


# Prediction-----------------------
model = tf.keras.models.load_model(os.path.join(path_models, 'LSTM_model.hdf5'))
#model.load_weights(os.path.join(path_models, 'weights'+'.h5'))

y_pred = model.predict(X_test)

y_pred_inverse = standardScaler.inverse_transform(y_pred)
y_test_inverse = standardScaler.inverse_transform(y_test)

print('MSE score:  {:.3f}'.format(mean_squared_error(y_pred_inverse, y_test_inverse)))
print('MAPE score: {:.3f}'.format(mean_absolute_percentage_error(y_pred_inverse, y_test_inverse)))