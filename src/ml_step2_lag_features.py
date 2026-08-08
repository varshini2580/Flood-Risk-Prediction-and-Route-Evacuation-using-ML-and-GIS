import pandas as pd
import config


def main():
	# Load master dataset
	df = pd.read_csv(str(config.processed("master_flood_dataset.csv")))

	# Convert date column
	df['date'] = pd.to_datetime(df['date'])

	# Create lag features
	df['rain_t1'] = df['rainfall_mm'].shift(1)
	df['rain_t2'] = df['rainfall_mm'].shift(2)
	df['runoff_t1'] = df['runoff_mm'].shift(1)
	df['soil_t1'] = df['soil_moisture'].shift(1)

	# Drop rows created with NaN due to lagging
	df = df.dropna()

	# Save lagged dataset
	df.to_csv(str(config.processed("ml_lagged_dataset.csv")), index=False)

	print("✅ Lag features created successfully")
	print(df.head())


if __name__ == "__main__":
	main()
