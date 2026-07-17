import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from math import sqrt
import joblib
import os
import logging

from hybrid_model import (
    create_hybrid_model,
    create_cnn_only_model,
    create_lstm_only_model,
    create_cnn_lstm_no_fusion_model
)
from SlidingWindow import create_sequences

SEQUENCE_LENGTH = 30
MAX_EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
LOG_FILE = 'ablation_study.log'
sensor_cols = [f'sensor_measurement_{i+1}' for i in range(21)]

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

print("Starting Data Preparation...")
logging.info("Starting Data Preparation...")

df_train = pd.read_csv('train_FD001.txt', sep=' ', header=None)
df_train.dropna(axis=1, how='all', inplace=True)
column_names = ['engine_id', 'cycle'] + [f'op_setting_{i+1}' for i in range(3)] + sensor_cols
df_train.columns = column_names
df_train.drop(columns=['op_setting_1', 'op_setting_2', 'op_setting_3'], inplace=True, errors='ignore')

scaler = joblib.load('scaler.pkl')
df_train[sensor_cols] = scaler.transform(df_train[sensor_cols])
X_seq_all, X_stats_all, y_all = create_sequences(df_train, SEQUENCE_LENGTH)

(X_seq_train, X_seq_val,
 X_stats_train, X_stats_val,
 y_train, y_val) = train_test_split(
    X_seq_all, X_stats_all, y_all, test_size=VALIDATION_SPLIT, random_state=42
)

df_test = pd.read_csv('test_FD001.txt', sep=' ', header=None)
df_test.dropna(axis=1, how='all', inplace=True)
df_test.columns = column_names
df_test.drop(columns=['op_setting_1', 'op_setting_2', 'op_setting_3'], inplace=True, errors='ignore')
df_test[sensor_cols] = scaler.transform(df_test[sensor_cols])
y_test = np.loadtxt('RUL_FD001.txt')

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

print("Data Preparation Complete.")
logging.info("Data Preparation Complete.")

MODELS_TO_RUN = {
    "CNN-only": create_cnn_only_model,
    "LSTM-only": create_lstm_only_model,
    "CNN-LSTM (no fusion)": create_cnn_lstm_no_fusion_model,
    "Full CNN-LSTM + Feature Fusion": create_hybrid_model
}

def evaluate_model_variant(model_name, model_fn):
    print(f"\n--- Running Ablation Model: {model_name} ---")
    logging.info(f"--- Running Ablation Model: {model_name} ---")

    input_shape = X_seq_train.shape[1:]
    num_stats = X_stats_train.shape[1]
    model_filepath = f"{model_name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')}.h5"

    if "Feature Fusion" in model_name:
        model = model_fn(input_shape, num_stats)
        train_inputs, val_inputs, test_inputs = [X_seq_train, X_stats_train], [X_seq_val, X_stats_val], [X_seq_test, X_stats_test]
    else:
        model = model_fn(input_shape)
        train_inputs, val_inputs, test_inputs = X_seq_train, X_seq_val, X_seq_test

    es_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
    mc_callback = ModelCheckpoint(model_filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=0)

    try:
        model.fit(
            train_inputs, y_train,
            validation_data=(val_inputs, y_val),
            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[es_callback, mc_callback], verbose=0
        )
        if "Feature Fusion" in model_name:
             best_model = model_fn(input_shape, num_stats)
        else:
             best_model = model_fn(input_shape)
        best_model.load_weights(model_filepath)

        y_pred = best_model.predict(test_inputs).flatten()
        mae, rmse, r2 = mean_absolute_error(y_test, y_pred), sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)
        model_size_mb = os.path.getsize(model_filepath) / (1024 * 1024)

        print(f" {model_name} Training Complete. Model saved to {model_filepath}")
        logging.info(f"{model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
        return mae, rmse, r2

    except Exception as e:
        print(f" Critical Error running {model_name}: {e}")
        logging.error(f"Critical Error running {model_name}: {e}")
        return "ERROR", "ERROR", "ERROR"

final_ablation_results = {}
for name, create_fn in MODELS_TO_RUN.items():
    mae, rmse, r2 = evaluate_model_variant(name, create_fn)
    final_ablation_results[name] = [mae, rmse, r2]

print("\n\n--- Final Ablation Study Results (Corrected) ---")
print("| Model Variant                  | MAE (cycles) | RMSE (cycles) | R²   |")
print("| :----------------------------- | :----------- | :------------ | :--- |")
for model, metrics in final_ablation_results.items():
    if "ERROR" in metrics:
        print(f"| {model:<30} | ERROR        | ERROR         | ERROR |")
    else:
        print(f"| {model:<30} | {metrics[0]:<12.2f} | {metrics[1]:<13.2f} | {metrics[2]:<4.2f} |")