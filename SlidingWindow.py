import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

# Example data
data = [[0, 0], [0, 0], [1, 1], [1, 1]]

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(data)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')


# Load and preprocess the dataset
column_names = ['engine_id', 'cycle'] + \
    [f'op_setting_{i+1}' for i in range(3)] + \
    [f'sensor_measurement_{i+1}' for i in range(21)]

df = pd.read_csv('train_FD001.txt', sep=' ', header=None)
df.dropna(axis=1, how='all', inplace=True)
df.columns = column_names

print(df.head())

# Create RUL column
rul_df = df.groupby('engine_id')['cycle'].max().reset_index()
rul_df.columns = ['engine_id', 'max_cycle']

df = df.merge(rul_df, on='engine_id')
df['RUL'] = df['max_cycle'] - df['cycle']
df.drop('max_cycle', axis=1, inplace=True)

print(df[['engine_id', 'cycle', 'RUL']].head())

# Normalize sensor data
sensor_cols = [col for col in df.columns if 'sensor_measurement' in col]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

print(df.head())

# Define create_sequences function
def create_sequences(df, sequence_length=30):
    X, y = [], []
    sensor_cols = [col for col in df.columns if 'sensor_measurement' in col]
    
    # Group by each engine
    for engine_id, engine_df in df.groupby('engine_id'):
        engine_df = engine_df.sort_values('cycle')
        features = engine_df[sensor_cols].values
        labels = engine_df['RUL'].values

        for i in range(len(engine_df) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(labels[i+sequence_length])  # target is next RUL

    return np.array(X), np.array(y)

# Call the function to generate sequences
X, y = create_sequences(df)

print("X shape:", X.shape)  # e.g. (num_samples, 30, num_features)
print("y shape:", y.shape)  # e.g. (num_samples,)
