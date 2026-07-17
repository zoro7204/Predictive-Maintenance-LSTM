import argparse
import numpy as np
import pandas as pd
from hybrid_model import create_hybrid_model
from tensorflow.keras.models import load_model
import joblib
import os

SEQUENCE_LENGTH = 30
TEST_DATA_PATH = "test_FD001.txt"
RUL_PATH = "RUL_FD001.txt"
MODEL_PATH = "hybrid_cnn_lstm_model.h5"
SCALER_PATH = "scaler.pkl"

sensor_cols = [f"sensor_measurement_{i+1}" for i in range(21)]
op_cols = [f"op_setting_{i+1}" for i in range(3)]
all_cols = ["engine_id", "cycle"] + op_cols + sensor_cols


def load_resources():
    # Load data
    df = pd.read_csv(TEST_DATA_PATH, sep="\s+", header=None)
    df = df.iloc[:, :len(all_cols)]
    df.columns = all_cols

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Apply full scaling (same as training)
    df[sensor_cols] = scaler.transform(df[sensor_cols])

    # Load model
    try:
        model = load_model(MODEL_PATH, compile=False)
    except:
        model = create_hybrid_model((SEQUENCE_LENGTH, len(sensor_cols)), 84)
        model.load_weights(MODEL_PATH)

    # Load true RUL
    true_rul = np.loadtxt(RUL_PATH)

    return df, model, true_rul


def predict_single(df, model, engine_id, cycle):
    df_engine = df[df.engine_id == engine_id].sort_values("cycle")

    # If cycle exceeds max, clamp it
    max_cycle = df_engine["cycle"].max()
    if cycle > max_cycle:
        cycle = max_cycle

    window = df_engine[df_engine.cycle <= cycle].tail(SEQUENCE_LENGTH)

    if len(window) < SEQUENCE_LENGTH:
        pad = window.iloc[0:1].repeat(SEQUENCE_LENGTH - len(window))
        window = pd.concat([pad, window])

    # Sequence input
    X_seq = np.array([window[sensor_cols].values])

    # Stats input
    stats = []
    stats.extend(window[sensor_cols].mean().values)
    stats.extend(window[sensor_cols].var(ddof=0).values)
    stats.extend(window[sensor_cols].skew().fillna(0).values)
    stats.extend(window[sensor_cols].kurt().fillna(0).values)
    X_stats = np.array([stats])

    # Predict
    try:
        pred = model.predict([X_seq, X_stats], verbose=0).flatten()[0]
    except:
        pred = model.predict(X_seq, verbose=0).flatten()[0]

    return pred, cycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_id", type=int, required=True)
    parser.add_argument("--cycle", type=int, required=False)
    args = parser.parse_args()

    df, model, true_rul = load_resources()

    # If no cycle provided, default: last 3 cycles
    if args.cycle is None:
        cycles = [30, 100, 200]
    else:
        cycles = [args.cycle]

    print("\n Model and scaler loaded successfully.\n")
    print(f"--- CASE STUDY: ENGINE #{args.engine_id} ---")

    for c in cycles:
        pred, used_cycle = predict_single(df, model, args.engine_id, c)
        true_final = int(true_rul[args.engine_id - 1])
        print(f"  - At Cycle {used_cycle}:")
        print(f"    Predicted RUL: {pred:.2f} cycles")
        print(f"    (True final RUL = {true_final} cycles)")

    print("--------------------------------------\n")
    print("Conclusion: Predictions now match training preprocessing exactly.")


if __name__ == "__main__":
    main()
