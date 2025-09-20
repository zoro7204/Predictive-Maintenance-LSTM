import pandas as pd # type: ignore

column_names = ['engine_id', 'cycle'] + \
    [f'op_setting_{i+1}' for i in range(3)] + \
    [f'sensor_measurement_{i+1}' for i in range(21)]

df = pd.read_csv('train_FD001.txt', sep=' ', header=None)
df.dropna(axis=1, how='all', inplace=True)
df.columns = column_names

print(df.head())

rul_df = df.groupby('engine_id')['cycle'].max().reset_index()
rul_df.columns = ['engine_id', 'max_cycle']

df = df.merge(rul_df, on='engine_id')
df['RUL'] = df['max_cycle'] - df['cycle']
df.drop('max_cycle', axis=1, inplace=True)

print(df[['engine_id', 'cycle', 'RUL']].head())

from sklearn.preprocessing import MinMaxScaler # type: ignore

sensor_cols = [col for col in df.columns if 'sensor_measurement' in col]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

print(df.head())

