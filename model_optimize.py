import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import pandas as pd
import os
from hybrid_model import create_hybrid_model 
from SlidingWindow import create_sequences

MODEL_PATH = 'Full_CNN-LSTM_plus_Feature_Fusion.h5'
TFLITE_MODEL_PATH = 'optimized_rul_model.tflite'
SEQUENCE_LENGTH = 30
sensor_cols = [f'sensor_measurement_{i+1}' for i in range(21)]
CUSTOM_OBJECTS = {'mse': MeanSquaredError(), 'mae': MeanAbsoluteError(), 'Adam': Adam}

print("Starting Data Preparation for Optimization...")
df_train = pd.read_csv('train_FD001.txt', sep=' ', header=None)
df_train.dropna(axis=1, how='all', inplace=True)
column_names = ['engine_id', 'cycle'] + [f'op_setting_{i+1}' for i in range(3)] + sensor_cols
df_train.columns = column_names
df_train.drop(columns=['op_setting_1', 'op_setting_2', 'op_setting_3'], inplace=True, errors='ignore')

scaler = joblib.load('scaler.pkl')
df_train[sensor_cols] = scaler.transform(df_train[sensor_cols])
X_seq_all, X_stats_all, y_all = create_sequences(df_train, SEQUENCE_LENGTH)
X_seq_train, X_seq_val, X_stats_train, X_stats_val, y_train, y_val = train_test_split(
    X_seq_all, X_stats_all, y_all, test_size=0.2, random_state=42
)
df_test = pd.read_csv('test_FD001.txt', sep=' ', header=None)
df_test.dropna(axis=1, how='all', inplace=True)
df_test.columns = column_names
df_test.drop(columns=['op_setting_1', 'op_setting_2', 'op_setting_3'], inplace=True, errors='ignore')
df_test[sensor_cols] = scaler.transform(df_test[sensor_cols])
X_seq_test_all, X_stats_test_all, y_test_all = create_sequences(df_test, SEQUENCE_LENGTH)
print("Data Preparation Complete.")

print("\n1. Rebuilding model structure and loading saved weights...")
input_shape = X_seq_train.shape[1:]
num_stats = X_stats_train.shape[1]
model = create_hybrid_model(input_shape, num_stats)
model.load_weights(MODEL_PATH)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae']) 

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,
    begin_step=2000,
    end_step=10000
)

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
pruned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print("2. Pruning wrappers applied. Starting fine-tuning...")

pruning_callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model.fit(
    [X_seq_train, X_stats_train], y_train,
    validation_data=([X_seq_val, X_stats_val], y_val),
    epochs=10, 
    batch_size=64,
    callbacks=pruning_callbacks,
    verbose=0
)
print("Fine-tuning complete.")

final_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
print("3. Pruning wrappers stripped.")

print("\n4. Applying 8-bit Post-Training Quantization (PTQ) with TFLite Fix...")

def representative_data_gen():
    """Generator for TFLite Converter to calibrate 8-bit values."""
    for i in range(100):
        yield [
            np.expand_dims(X_seq_test_all[i], axis=0).astype(np.float32),
            np.expand_dims(X_stats_test_all[i], axis=0).astype(np.float32)
        ]

converter = tf.lite.TFLiteConverter.from_keras_model(final_pruned_model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] 
converter.target_spec.supported_types = [tf.float16]
converter._experimental_lower_tensor_list_ops = False

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

converter.representative_dataset = representative_data_gen

tflite_model = converter.convert()
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

optimized_size_mb = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
print(f"\n Optimization Complete. TFLite Model Saved to: {TFLITE_MODEL_PATH}")
print(f"Final Optimized Model Size: {optimized_size_mb:.3f} MB")

if optimized_size_mb <= 2.3:
    print(" EDGE DEPLOYMENT TARGET MET!")
else:
    print(" WARNING: Final size exceeds 2.3 MB. Final performance evaluation needed.")