import requests
import pandas as pd
import time
import config


def main():
    centroids = pd.read_csv(str(config.processed("grids_centroid.csv")))

    start_date = "20210902"
    end_date = "20220802"

    all_data = []

    for index, row in centroids.iterrows():
        zone = row["zone_id"]
        lat = row["latitude"]
        lon = row["longitude"]

        print(f"Downloading Zone {zone}...")

        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"

        response = requests.get(url)

        try:
            data = response.json()
            rainfall = data["properties"]["parameter"]["PRECTOTCORR"]

            for date, value in rainfall.items():
                all_data.append({
                    "zone_id": zone,
                    "date": date,
                    "rainfall_mm": value
                })

        except Exception as e:
            print(f"Error in Zone {zone}")
            print(data)
            continue

        time.sleep(1)

    df = pd.DataFrame(all_data)
    df.to_csv(str(config.processed("zone_daily_rainfall.csv")), index=False)

    print("Rainfall data downloaded successfully!")


if __name__ == "__main__":
    main()