import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from math import sqrt
import argparse
import logging
from SlidingWindow import scaler, sensor_cols  # Reuse the fitted scaler
import os

# -------------------- Parameters --------------------
SEQUENCE_LENGTH = 30
MODEL_PATH = 'lstm_model.h5'  # or 'gru_model.h5'
DATA_PATH = 'test_FD001.txt'
RUL_PATH = 'RUL_FD001.txt'
LOG_FILE = 'predictive_maintenance.log'

# -------------------- Logging Configuration --------------------
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- Load Model --------------------
try:
    # Define custom objects if needed
    custom_objects = {
        'MeanSquaredError': MeanSquaredError,
        'mse': MeanSquaredError()  # Explicitly define 'mse'
    }
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    logging.info(f"Loaded model: {MODEL_PATH}")
    print(f"✅ Loaded model: {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# -------------------- Load and Preprocess Test Data --------------------
column_names = ['engine_id', 'cycle'] + \
               [f'op_setting_{i+1}' for i in range(3)] + \
               [f'sensor_measurement_{i+1}' for i in range(21)]

try:
    df = pd.read_csv(DATA_PATH, sep=' ', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = column_names
    logging.info(f"Loaded and preprocessed data from {DATA_PATH}")
    print(f"✅ Loaded and preprocessed data from {DATA_PATH}")
except FileNotFoundError:
    logging.error(f"Data file not found: {DATA_PATH}")
    raise
except Exception as e:
    logging.error(f"Error processing data: {e}")
    raise

# Normalize with same scaler as training
df[sensor_cols] = scaler.transform(df[sensor_cols])

# -------------------- Load True RUL Values --------------------
try:
    rul_truth = pd.read_csv(RUL_PATH, header=None)
    rul_truth.columns = ['RUL']
    rul_truth['engine_id'] = rul_truth.index + 1
    logging.info(f"Loaded true RUL values from {RUL_PATH}")
    print(f"✅ Loaded true RUL values from {RUL_PATH}")
except FileNotFoundError:
    logging.error(f"RUL file not found: {RUL_PATH}")
    raise
except Exception as e:
    logging.error(f"Error processing RUL data: {e}")
    raise

# -------------------- Create Sequences --------------------

def make_last_sequences(df, sequence_length):
    """Generate sequences using the last window of each engine's data."""
    X_list = []
    true_rul_list = []

    for engine_id, engine_df in df.groupby('engine_id'):
        engine_df = engine_df.sort_values('cycle')
        if len(engine_df) >= sequence_length:
            last_window = engine_df[-sequence_length:]
            features = last_window[sensor_cols].values
            X_list.append(features)

            # Get true RUL from the file
            true_rul = rul_truth.loc[rul_truth['engine_id'] == engine_id, 'RUL'].values[0]
            true_rul_list.append(true_rul)

    return np.array(X_list), np.array(true_rul_list)

def make_full_sequences(df, sequence_length):
    """Generate all possible sliding sequences from the entire dataset."""
    X_list = []
    true_rul_list = []

    for engine_id, engine_df in df.groupby('engine_id'):
        engine_df = engine_df.sort_values('cycle')
        if len(engine_df) >= sequence_length:
            for i in range(len(engine_df) - sequence_length):
                window = engine_df.iloc[i:i+sequence_length]
                features = window[sensor_cols].values
                X_list.append(features)

                # Get true RUL from the file
                true_rul = rul_truth.loc[rul_truth['engine_id'] == engine_id, 'RUL'].values[0]
                true_rul_list.append(true_rul)

    return np.array(X_list), np.array(true_rul_list)

# -------------------- Argument Parsing --------------------
parser = argparse.ArgumentParser(description='Evaluate Predictive Maintenance Model')
parser.add_argument('--mode', type=str, choices=['final', 'full'], default='final',
                    help='Evaluation mode: "final" for last sequence or "full" for all sequences.')
args = parser.parse_args()

# -------------------- Generate Sequences --------------------
if args.mode == 'final':
    logging.info(f"Using {args.mode} evaluation mode (last window per engine)...")
    print(f"✅ Using {args.mode} evaluation mode (last window per engine)...")
    X_test, y_test = make_last_sequences(df, SEQUENCE_LENGTH)
elif args.mode == 'full':
    logging.info(f"Using {args.mode} evaluation mode (all possible sequences)...")
    print(f"✅ Using {args.mode} evaluation mode (all possible sequences)...")
    X_test, y_test = make_full_sequences(df, SEQUENCE_LENGTH)

logging.info(f"Created {X_test.shape[0]} test sequences of shape {X_test.shape[1:]}")
print(f"✅ Created {X_test.shape[0]} test sequences of shape {X_test.shape[1:]}")

# -------------------- Predict RUL --------------------
try:
    y_pred = model.predict(X_test).flatten()
    logging.info("Predictions generated successfully.")
    print(f"✅ Predictions generated successfully.")
except Exception as e:
    logging.error(f"Error generating predictions: {e}")
    raise

# -------------------- Evaluate --------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

logging.info(f"Model Evaluation on Test Set: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R2 = {r2:.2f}")
print(f"\n📊 Model Evaluation on Test Set:")
print(f"MAE  = {mae:.2f} cycles")
print(f"RMSE = {rmse:.2f} cycles")
print(f"R2   = {r2:.2f}")

# -------------------- Plot --------------------
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True RUL', marker='o')
plt.plot(y_pred, label='Predicted RUL', marker='x')
plt.xlabel("Engine Index")
plt.ylabel("RUL")
plt.title(f"True vs Predicted RUL ({args.mode.capitalize()} Mode)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'rul_plot_{args.mode}.png')
plt.show()

# -------------------- Unit Tests --------------------
def test_make_last_sequences():
    print("\n🔍 Running test_make_last_sequences()...")

    # Create mock data with correct number of sensor columns
    num_sensors = len(sensor_cols)
    data = {
        'engine_id': [1]*35 + [2]*35,
        'cycle': list(range(1, 36)) * 2
    }

    for i, sensor in enumerate(sensor_cols):
        data[sensor] = [0.1 + i * 0.01] * 70  # constant dummy values

    df_test = pd.DataFrame(data)

    # Normalize using training scaler
    df_test[sensor_cols] = scaler.transform(df_test[sensor_cols])

    # Create mock RUL truth
    rul_test = pd.DataFrame({
        'engine_id': [1, 2],
        'RUL': [112, 98]
    })
    global rul_truth
    rul_truth = rul_test

    # Call function
    sequences, rul = make_last_sequences(df_test, SEQUENCE_LENGTH)

    assert sequences.shape[0] == 2, f"Expected 2 sequences, got {sequences.shape[0]}"
    assert sequences.shape[1] == SEQUENCE_LENGTH, f"Expected sequence length {SEQUENCE_LENGTH}, got {sequences.shape[1]}"
    assert sequences.shape[2] == num_sensors, f"Expected {num_sensors} features, got {sequences.shape[2]}"
    assert len(rul) == 2, f"Expected 2 RUL values, got {len(rul)}"

    print("✅ test_make_last_sequences passed.")


def test_make_full_sequences():
    print("\n🔍 Running test_make_full_sequences()...")

    # Create mock data
    num_sensors = len(sensor_cols)
    data = {
        'engine_id': [1]*35 + [2]*35,
        'cycle': list(range(1, 36)) * 2
    }

    for i, sensor in enumerate(sensor_cols):
        data[sensor] = [0.1 + i * 0.01] * 70

    df_test = pd.DataFrame(data)

    # Normalize
    df_test[sensor_cols] = scaler.transform(df_test[sensor_cols])

    # Mock RUL
    rul_test = pd.DataFrame({
        'engine_id': [1, 2],
        'RUL': [100, 90]
    })
    global rul_truth
    rul_truth = rul_test

    # Run function
    sequences, rul = make_full_sequences(df_test, SEQUENCE_LENGTH)

    expected_sequences = (35 - SEQUENCE_LENGTH) * 2
    assert sequences.shape[0] == expected_sequences, f"Expected {expected_sequences} sequences, got {sequences.shape[0]}"
    assert sequences.shape[1:] == (SEQUENCE_LENGTH, num_sensors), f"Unexpected sequence shape {sequences.shape[1:]}"
    assert rul.shape[0] == expected_sequences, f"Expected {expected_sequences} RUL values, got {rul.shape[0]}"

    print("✅ test_make_full_sequences passed.")

# Run unit tests
test_make_last_sequences()
test_make_full_sequences()
