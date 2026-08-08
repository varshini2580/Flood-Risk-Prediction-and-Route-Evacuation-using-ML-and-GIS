import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import config


def main():
    df = pd.read_csv(str(config.processed("ml_lagged_dataset.csv")))

    # Features and target
    X = df.drop(columns=["date", "runoff_mm"])
    y = df["runoff_mm"]

    # Train-test split (time-aware)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Save final predictions (date + predicted runoff) as CSV
    test_dates = df["date"][split:]
    pred_df = pd.DataFrame({
        "date": test_dates,
        "predicted_runoff_mm": y_pred
    })

    pred_df.to_csv(str(config.results("xgboost_predictions.csv")), index=False)

    # Calculate flood threshold
    threshold = pred_df["predicted_runoff_mm"].quantile(0.90)
    print("Flood Threshold:", threshold)

    pred_df["flood_label"] = (pred_df["predicted_runoff_mm"] >= threshold).astype(int)

    pred_df.to_csv(str(config.processed("final_flood_prediction.csv")), index=False)
    print("✅ Final flood prediction CSV saved")

    # Metrics
    # ==========================
    # REGRESSION METRICS
    # ==========================

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n========== Regression Metrics ==========")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # ==========================
    # CLASSIFICATION METRICS
    # ==========================

    # Use the same threshold used for flood prediction
    threshold = pred_df["predicted_runoff_mm"].quantile(0.90)

    # Convert runoff values to flood/non-flood labels
    y_true_class = (y_test >= threshold).astype(int)
    y_pred_class = (y_pred >= threshold).astype(int)

    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, zero_division=0)

    print("\n========== Classification Metrics ==========")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true_class, y_pred_class))

    print("\nClassification Report")
    print(classification_report(y_true_class, y_pred_class))

    # Save all metrics
    metrics_df = pd.DataFrame({
        "Metric": [
            "RMSE",
            "MAE",
            "R²",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Value": [
            rmse,
            mae,
            r2,
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    metrics_df.to_csv(str(config.results("evaluation_metrics.csv")), index=False)

    # Plot Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true_class,
        y_pred_class,
        cmap="Blues"
    )

    plt.title("Flood Classification Confusion Matrix")
    plt.show()

    # Plot prediction
    plt.figure(figsize=(12,5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Flood Runoff Prediction using XGBoost")
    plt.legend()
    plt.show()

    # Feature importance
    importance = model.feature_importances_
    features = X.columns

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    imp_df.to_csv(str(config.results("xgboost_feature_importance.csv")), index=False)

    # Plot importance
    plt.figure(figsize=(8,5))
    plt.barh(imp_df["Feature"], imp_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("XGBoost Feature Importance")
    plt.show()


if __name__ == "__main__":
    main()
