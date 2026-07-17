import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from math import sqrt

import logging
import joblib

from hybrid_model import create_hybrid_model

SEQUENCE_LENGTH = 30
MODEL_PATH = 'hybrid_cnn_lstm_model.h5' 
DATA_PATH = 'test_FD001.txt'
RUL_PATH = 'RUL_FD001.txt'
LOG_FILE = 'predictive_maintenance.log'
sensor_cols = [f'sensor_measurement_{i+1}' for i in range(21)]

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    scaler = joblib.load('scaler.pkl')
    print("Loaded scaler from scaler.pkl")
except FileNotFoundError:
    print("Error: scaler.pkl not found. Please run model_train.py first.")
    raise

print("\n Loading and preprocessing test data...")
df_test = pd.read_csv(DATA_PATH, sep=' ', header=None)
df_test.dropna(axis=1, how='all', inplace=True)
column_names = ['engine_id', 'cycle'] + [f'op_setting_{i+1}' for i in range(3)] + sensor_cols
df_test.columns = column_names
df_test.drop(columns=['op_setting_1', 'op_setting_2', 'op_setting_3'], inplace=True, errors='ignore')
df_test[sensor_cols] = scaler.transform(df_test[sensor_cols])

y_test = np.loadtxt(RUL_PATH)

def create_test_sequences(df, sequence_length):
    X_seq_list, X_stats_list = [], []
    for engine_id in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine_id]
        window = engine_df.iloc[-sequence_length:]
        
        X_seq_list.append(window[sensor_cols].values)
        
        stats_features = [window[sensor_col].mean() for sensor_col in sensor_cols]
        stats_features.extend([window[sensor_col].var() for sensor_col in sensor_cols])
        stats_features.extend([window[sensor_col].skew() for sensor_col in sensor_cols])
        stats_features.extend([window[sensor_col].kurt() for sensor_col in sensor_cols])
        X_stats_list.append(stats_features)
        
    return np.array(X_seq_list), np.array(X_stats_list)

X_seq_test, X_stats_test = create_test_sequences(df_test, SEQUENCE_LENGTH)

print(f"Created {X_seq_test.shape[0]} test sequences of shape {X_seq_test.shape[1:]}")
print(f"Created {X_stats_test.shape[0]} test statistical features of shape {X_stats_test.shape[1:]}")

print("\n Rebuilding model structure and loading weights...")
try:
    input_shape = X_seq_test.shape[1:]
    num_stats_features = X_stats_test.shape[1]
    model = create_hybrid_model(input_shape, num_stats_features)
    model.load_weights(MODEL_PATH)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    print(f" Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f" Error loading model weights: {e}")
    raise

print("\n Generating predictions...")
y_pred = model.predict([X_seq_test, X_stats_test]).flatten()
print(f" Predictions generated successfully.")

mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n\n=============================================")
print(f" FINAL MODEL EVALUATION RESULTS")
print(f"=============================================")
print(f"MAE  = {mae:.2f} cycles")
print(f"RMSE = {rmse:.2f} cycles")
print(f"R²   = {r2:.2f}")
print(f"=============================================\n")

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True RUL', color='blue', alpha=0.7, marker='o', linestyle='-')
plt.plot(y_pred, label='Predicted RUL', color='red', marker='x', linestyle='--')
plt.title('True vs. Predicted RUL for Test Set')
plt.xlabel('Engine ID (Test Unit)')
plt.ylabel('Remaining Useful Life (Cycles)')
plt.legend()
plt.grid(True)
plt.savefig('rul_plot_final_evaluation.png')
print(" Final evaluation plot saved to 'rul_plot_final_evaluation.png'")