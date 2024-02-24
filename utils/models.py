from keras.models import Model

import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
from utils.config_reader import config_reader
# Import parameters
config = config_reader('../config/config.json')

class ModelLSTM(Model):
    """Класс создаёт модель LSTM, наследуя класс от tf.keras.
    Параметры:
    ----------
    n_timesteps (_int_) - кол-во временных периодов
    n_channels (_int_) - кол-во датчиков
    output_units - конечный слой модели
    lstm_units - размерность модели из конфига
    """
    def __init__(self, data):
    
        super().__init__()
        # ------- параметры ------------
        _, self.n_timesteps, self.n_channels = data.shape
        
        # -------- слои модели ----------------
        self.input_channels = x = tf.keras.layers.Input(shape=(self.n_timesteps, self.n_channels))# #shape=(self.n_timesteps, self.n_channels)
      
        x = tf.keras.layers.LSTM(units=256, return_sequences=True, activation = 'relu')(x) #
        x = tf.keras.layers.LSTM(units=32)(x) #, return_sequences=True, dropout=0.5
        #x = tf.keras.layers.LSTM(units=7)(x) #, dropout=0.5
        #x = tf.keras.layers.Dense(1, activation='relu')(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        #x = tf.keras.layers.Dropout(0.5)(x)

        self.output_channels = tf.keras.layers.Dense(units=1)(x)
        
        print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_channels}")

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
    
    # def compile_model(model):
    #     """Compile the regression model
    #     """
        
    #     optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    #     model.compile(loss = 'mse', metrics = ['mae', 'mse'], optimizer = optimizer)
        
    #     return model 
    
