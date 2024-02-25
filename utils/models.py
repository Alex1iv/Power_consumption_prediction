import numpy as np
import pandas as pd
from keras.models import Model

import tensorflow as tf
# from keras.models import Sequential
from keras.layers import LSTM, Dense
from utils.config_reader import config_reader
# Import parameters
config = config_reader('../config/config.json')

class ModelLSTM(Model):
    """ Model inherits the LSTM class from Keras.
    Параметры:
    ----------
    Model - data
   
    """
    def __init__(self, data):
    
        super().__init__()
        # ------- Parameters ------------
        _, self.n_timesteps, self.n_channels = data.shape
        
        # -------- Model layers ----------------
        self.input_channels = x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels)) 
      
        x = tf.keras.layers.LSTM(units=256, return_sequences=True, activation = 'relu')(x) #
        x = tf.keras.layers.LSTM(units=32)(x) #, return_sequences=True, dropout=0.5
        #x = tf.keras.layers.LSTM(units=7)(x) #, dropout=0.5
        #x = tf.keras.layers.Dense(1, activation='relu')(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        #x = tf.keras.layers.Dropout(0.5)(x)

        self.output_channels = tf.keras.layers.Dense(units=1)(x)
        
        print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {(self.output_channels)}")

    def build_model(self):
        """Initiate the model
        """
        model = tf.keras.Model(
            inputs=self.input_channels,
            outputs=self.output_channels,
            name="model_LSTM"
        )        
        model.summary()
        
        #--------------
        # compile model
        #--------------
        optimizer = tf.keras.optimizers.Adam(learning_rate = config.model.learning_rate)
        model.compile(loss = 'mse', metrics = ['mae', 'mse'], optimizer = optimizer)

        return model
    

    
class linear_model(Model):
    """Linear mode class
    Params:
    ----------
    n_timesteps (_int_) - кол-во временных периодов
    n_channels (_int_) - кол-во датчиков
    output_units - конечный слой модели
    lstm_units - размерность модели из конфига
    """
    def __init__(self, data):
    
        super().__init__()
        # ------- Parameters ------------
        _, self.n_timesteps, self.n_channels = data.shape
        
        # -------- Model layers ----------------
        self.input_channels = x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels))
      
        x = Dense(units=256,  activation = 'relu')(x)
        x = Dense(units=128)(x)
        x = Dense(units=64)(x)
        self.output_channels = tf.keras.layers.Dense(units=1)(x)
        
        print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_channels}")

    def build_model(self):
        """Initiate the model
        """
        model = tf.keras.Model(
            inputs=self.input_channels,
            outputs=self.output_channels,
            name="linear_model"
        )        
        model.summary()
        
        # compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate = config.model.learning_rate)
        model.compile(loss = 'mse', metrics = ['mae', 'mse'], optimizer = optimizer)

        return model
    


def get_predictions(periods_to_forecast:int, last_prediction:np.array, last_index:pd.Timestamp, scaler, model, n_steps=config.n_steps, n_features:int=1)->pd.Series:
    """Generate predictions for future periods

    Args:
        periods_to_forecast (int): _description_
        data (np.array): _description_
        n_features (int, optional): _description_. Defaults to 1.

    Returns:
        pd.Series: _description_
    """
    future_indexes = []
    predicted_arr = np.array([], dtype='float32')

    for i in range(periods_to_forecast):
        # join new indexes
        future_indexes.append(last_index + pd.Timedelta(days=i)) 
        
        last_prediction = last_prediction.reshape((1, n_steps, n_features))
        
        yhat = model.predict(last_prediction, verbose=0)
         
        predicted_arr = np.append(predicted_arr, yhat.squeeze())
       
        # append and roll the prediction
        last_prediction = np.vstack((last_prediction.reshape(n_steps,-1), yhat))[1:].ravel()
    
    predictions = pd.Series(scaler.inverse_transform(predicted_arr.reshape(-1,1)).squeeze(), name='future', index=future_indexes)
    
    return predictions
