import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, GRU, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from SlidingWindow import create_sequences, df  # Reuse your sliding window logic

# Generate sequence data
X, y = create_sequences(df, sequence_length=30)

# Split data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model builder function
def build_model(model_type='lstm', input_shape=(30, X.shape[2])):
    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    elif model_type == 'gru':
        model.add(GRU(64, return_sequences=False, input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output: predicted RUL
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Train and evaluate both models
histories = {}
for model_type in ['lstm', 'gru']:
    print(f"\nTraining {model_type.upper()} model...")
    model = build_model(model_type)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        verbose=1
    )
    histories[model_type] = history

# Plot training history
plt.figure(figsize=(12, 5))
for model_type in histories:
    plt.plot(histories[model_type].history['val_loss'], label=f'{model_type.upper()} Validation Loss')
plt.title("Model Comparison: Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save best-performing model (choose LSTM or GRU based on val loss)
best_model_type = min(histories, key=lambda k: min(histories[k].history['val_loss']))
best_model = build_model(best_model_type)

# Save to disk
best_model.save(f'{best_model_type}_model.h5')
print(f"Saved best model: {best_model_type}_model.h5")

# Save model
model.save(f'{model_type}_model.h5')


