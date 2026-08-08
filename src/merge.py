import pandas as pd
import numpy as np
import config


def main():
    # ----------------------------
    # 1. Load datasets
    # ----------------------------
    rain = pd.read_csv(str(config.processed("zone_daily_rainfall.csv")))
    soil = pd.read_csv(str(config.processed("zone_daily_soil_moisture.csv")))
    hand = pd.read_csv(str(config.processed("zone_hand_ml.csv")))

    # Ensure date format consistent
    rain["date"] = rain["date"].astype(str)
    soil["date"] = soil["date"].astype(str)

    # ----------------------------
    # 2. Merge rainfall + soil moisture
    # ----------------------------
    df = pd.merge(rain, soil, on=["zone_id", "date"])

    # ----------------------------
    # 3. Merge HAND (spatial vulnerability)
    # ----------------------------
    df = pd.merge(df, hand[["zone_id", "hf_mean"]], on="zone_id")

    print("After merge:", df.shape)

    # ----------------------------
    # 4. Compute Runoff
    # ----------------------------
    df["runoff_mm"] = df["rainfall_mm"] * df["soil_moisture"]

    # ----------------------------
    # 5. Normalize HAND
    # ----------------------------
    df["hand_norm"] = (df["hf_mean"] - df["hf_mean"].min()) / \
                      (df["hf_mean"].max() - df["hf_mean"].min())

    # Invert (lower HAND = higher vulnerability)
    df["vulnerability"] = 1 - df["hand_norm"]

    # ----------------------------
    # 6. Spatial Flood Risk Index
    # ----------------------------
    df["flood_risk_index"] = df["runoff_mm"] * df["vulnerability"]

    # ----------------------------
    # 7. Create Flood Label (Top 15% risk days)
    # ----------------------------
    threshold = df["flood_risk_index"].quantile(0.85)
    df["flood_label"] = (df["flood_risk_index"] > threshold).astype(int)

    df["zone_id"] = df["zone_id"].astype(int)

    # ----------------------------
    # 8. Flood Days Per Zone
    # ----------------------------
    print("\nFlood days per zone:")
    flood_days = df.groupby("zone_id")["flood_label"].sum()
    print(flood_days)

    # ----------------------------
    # 9. Flood Frequency Per Zone
    # ----------------------------
    zone_summary = flood_days.reset_index()
    zone_summary.columns = ["zone_id", "flood_days"]
    zone_summary["flood_frequency"] = zone_summary["flood_days"] / 335

    print("\nZone Flood Frequency:")
    print(zone_summary)

    # ----------------------------
    # 10. Save Outputs
    # ----------------------------
    df.to_csv(str(config.processed("zone_ml_spatial_final_dataset.csv")), index=False)
    zone_summary.to_csv(str(config.processed("zone_flood_summary.csv")), index=False)

    print("\nSpatial dataset saved successfully!")
    print("\nZone summary saved successfully!")
    print(df.head())


if __name__ == "__main__":
    main()