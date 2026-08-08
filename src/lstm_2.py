from pathlib import Path
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Processed data folder
PROCESSED_DIR = ROOT / "data" / "processed"
# -----------------------------
# 1. LOAD PREPARED DATA
# -----------------------------
X_train = np.load(PROCESSED_DIR / "X_train.npy")
X_test = np.load(PROCESSED_DIR / "X_test.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# -----------------------------
# 2. BUILD LSTM MODEL
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# -----------------------------
# 3. TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# -----------------------------
# 4. PREDICTIONS
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 5. EVALUATION
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)

# -----------------------------
# 6. PLOT RESULTS
# -----------------------------
plt.figure()
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Flood Runoff Prediction using LSTM")
plt.show()

# -----------------------------
# 7. SAVE MODEL
# -----------------------------
model.save("flood_lstm_model.h5")

print("✅ LSTM training and prediction completed successfully")
