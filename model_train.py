import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from hybrid_model import create_hybrid_model
from SlidingWindow import create_sequences

column_names = ['engine_id', 'cycle'] + \
    [f'op_setting_{i+1}' for i in range(3)] + \
    [f'sensor_measurement_{i+1}' for i in range(21)]
df_train = pd.read_csv('train_FD001.txt', sep=' ', header=None)
df_train.dropna(axis=1, how='all', inplace=True)
df_train.columns = column_names

rul_df = df_train.groupby('engine_id')['cycle'].max().reset_index()
rul_df.columns = ['engine_id', 'max_cycle']
df_train = df_train.merge(rul_df, on='engine_id')
df_train.drop('max_cycle', axis=1, inplace=True)

sensor_cols = [col for col in df_train.columns if 'sensor_measurement' in col]
scaler = MinMaxScaler()
df_train[sensor_cols] = scaler.fit_transform(df_train[sensor_cols])
joblib.dump(scaler, 'scaler.pkl')

print("\nData Preprocessing Complete.")

SEQUENCE_LENGTH = 30
X_seq, X_stats, y = create_sequences(df_train, SEQUENCE_LENGTH)

print(f"\nCreated {X_seq.shape[0]} sequences.")
print("X_seq shape (for CNN-LSTM):", X_seq.shape)
print("X_stats shape (for Fusion):", X_stats.shape)
print("y shape (for RUL):", y.shape)

X_seq_train, X_seq_val, X_stats_train, X_stats_val, y_train, y_val = train_test_split(
    X_seq, X_stats, y, test_size=0.2, random_state=42
)

num_stats_features = X_stats_train.shape[1]
input_shape = (SEQUENCE_LENGTH, X_seq_train.shape[2])

model = create_hybrid_model(input_shape, num_stats_features)
print(model.summary())

history = model.fit(
    [X_seq_train, X_stats_train], y_train,
    validation_data=([X_seq_val, X_stats_val], y_val),
    epochs=50,
    batch_size=64,
    verbose=1
)

model.save('hybrid_cnn_lstm_model.h5')
print("\n Saved hybrid_cnn_lstm_model.h5")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Hybrid Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()