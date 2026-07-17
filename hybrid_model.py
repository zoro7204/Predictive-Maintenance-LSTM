import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Reshape 

def create_hybrid_model(input_shape, num_stats_features):
    cnn_lstm_input = Input(shape=input_shape)
    conv = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_lstm_input)
    pool = MaxPooling1D(pool_size=2)(conv)
    lstm = LSTM(64)(pool)

    stats_input = Input(shape=(num_stats_features,))

    fused_features = Concatenate()([lstm, stats_input])
    
    dense1 = Dense(32, activation='relu')(fused_features)
    output = Dense(1, activation='linear')(dense1)

    model = Model(inputs=[cnn_lstm_input, stats_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

def create_cnn_only_model(input_shape):
    """Creates a CNN-only model for RUL prediction."""
    cnn_input = Input(shape=input_shape, name='cnn_input')
    conv = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_input)
    pool = MaxPooling1D(pool_size=2)(conv)
    flat = Flatten()(pool) 
    
    dense1 = Dense(32, activation='relu')(flat)
    output = Dense(1, activation='linear', name='output')(dense1)

    model = Model(inputs=cnn_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

def create_lstm_only_model(input_shape):
    """Creates an LSTM-only model for RUL prediction."""
    lstm_input = Input(shape=input_shape, name='lstm_input')
    lstm = LSTM(64)(lstm_input)
    
    dense1 = Dense(32, activation='relu')(lstm)
    output = Dense(1, activation='linear', name='output')(dense1)

    model = Model(inputs=lstm_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

def create_cnn_lstm_no_fusion_model(input_shape):
    """Creates a CNN-LSTM model without the statistical feature fusion branch."""
    cnn_lstm_input = Input(shape=input_shape, name='cnn_lstm_input')
    conv = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_lstm_input)
    pool = MaxPooling1D(pool_size=2)(conv)
    lstm = LSTM(64)(pool)

    dense1 = Dense(32, activation='relu')(lstm)
    output = Dense(1, activation='linear', name='output')(dense1)

    model = Model(inputs=cnn_lstm_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model