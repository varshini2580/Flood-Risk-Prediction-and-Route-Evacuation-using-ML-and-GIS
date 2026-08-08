import pandas as pd
import config


def main():
	# Load ML dataset
	ml = pd.read_csv(str(config.processed("final_flood_prediction.csv")))  # change if file name different

	# Load zones dataset
	zones = pd.read_csv(str(config.processed("zones.csv")))

	# Keep only required columns
	zones = zones[['zone_id', 'zone_name']]

	# Create cross join (Cartesian product)
	ml['key'] = 1
	zones['key'] = 1

	final = pd.merge(ml, zones, on='key').drop('key', axis=1)

	# Save final dataset
	final.to_csv(str(config.processed("final_zone_ml_dataset.csv")), index=False)

	print("Done!")
	print("Total rows created:", len(final))


if __name__ == "__main__":
	main()