import pandas as pd
import config


def main():
	# Load CSVs
	rain = pd.read_csv(str(config.raw("rainfall.csv")))
	runoff = pd.read_csv(str(config.raw("runoff.csv")))
	soil = pd.read_csv(str(config.raw("soil_moisture.csv")))

	# Convert date column to datetime
	rain['date'] = pd.to_datetime(rain['date'], format='%d-%m-%Y', errors='coerce')
	runoff['date'] = pd.to_datetime(runoff['Row Labels'], format='%d-%m-%Y', errors='coerce')
	soil['date'] = pd.to_datetime(soil['daily_date'], format='%Y-%m-%d', errors='coerce')

	# Drop rows with invalid dates
	rain = rain.dropna(subset=['date'])
	runoff = runoff.dropna(subset=['date'])
	soil = soil.dropna(subset=['date'])

	# Rename columns (adjust if names differ)
	rain = rain[['date', 'rainfall_mm']]
	runoff = runoff[['date', 'Average of runoff_mm']].rename(columns={'Average of runoff_mm': 'runoff_mm'})
	soil = soil[['date', 'soil_moisture']]

	# Merge step-by-step on date
	df = rain.merge(runoff, on='date', how='inner')
	df = df.merge(soil, on='date', how='inner')

	# Sort by date
	df = df.sort_values('date')

	# Fill missing values
	df['rainfall_mm'] = df['rainfall_mm'].fillna(0)
	df['runoff_mm'] = df['runoff_mm'].ffill()
	df['soil_moisture'] = df['soil_moisture'].ffill()

	# Save master CSV
	df.to_csv(str(config.processed("master_flood_dataset.csv")), index=False)

	print("✅ Master CSV created successfully!")
	print(df.head())


if __name__ == "__main__":
	main()
