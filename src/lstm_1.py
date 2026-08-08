import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import config


def create_sequences(X, y, window=7):
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i-window:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def main():
    # -----------------------------
    # 1. LOAD MASTER CSV
    # -----------------------------
    df = pd.read_csv(str(config.processed("master_flood_dataset.csv")))   # <-- change name if needed

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date (IMPORTANT for LSTM)
    df = df.sort_values('date').reset_index(drop=True)

    # -----------------------------
    # 2. SELECT FEATURES & TARGET
    # -----------------------------
    features = ['rainfall_mm', 'soil_moisture']
    target = 'runoff_mm'

    # -----------------------------
    # 3. SCALE DATA
    # -----------------------------
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[[target]])

    # -----------------------------
    # 4. CREATE SEQUENCES (LSTM WINDOW)
    # -----------------------------
    WINDOW_SIZE = 7   # 7-day lookback
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW_SIZE)

    print("LSTM Input Shape:", X_seq.shape)
    print("LSTM Output Shape:", y_seq.shape)

    # -----------------------------
    # 5. TRAIN–TEST SPLIT (NO SHUFFLE)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # -----------------------------
    # 6. SAVE FOR MODEL TRAINING
    # -----------------------------
    np.save(str(config.processed("X_train.npy")), X_train)
    np.save(str(config.processed("X_test.npy")), X_test)
    np.save(str(config.processed("y_train.npy")), y_train)
    np.save(str(config.processed("y_test.npy")), y_test)

    print("✅ LSTM data preparation completed successfully")


if __name__ == "__main__":
    main()
