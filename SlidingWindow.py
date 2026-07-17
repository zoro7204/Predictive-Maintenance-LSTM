import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

sensor_cols = [f'sensor_measurement_{i+1}' for i in range(21)]

def create_sequences(df, sequence_length):
    X_seq_list, X_stats_list, y_list = [], [], []
    
    for engine_id in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine_id].copy()

        max_cycle = engine_df['cycle'].max()
        engine_df['RUL'] = max_cycle - engine_df['cycle']

        features = engine_df[sensor_cols].values

        for i in range(len(features) - sequence_length + 1):
            window = engine_df.iloc[i:i + sequence_length]

            X_seq_list.append(window[sensor_cols].values)

            stats_features = [
                window[sensor_col].mean() for sensor_col in sensor_cols
            ]
            stats_features.extend([
                window[sensor_col].var() for sensor_col in sensor_cols
            ])
            stats_features.extend([
                window[sensor_col].skew() for sensor_col in sensor_cols
            ])
            stats_features.extend([
                window[sensor_col].kurt() for sensor_col in sensor_cols
            ])
            X_stats_list.append(stats_features)

            y_list.append(engine_df['RUL'].iloc[i + sequence_length - 1])
            
    return np.array(X_seq_list), np.array(X_stats_list), np.array(y_list)