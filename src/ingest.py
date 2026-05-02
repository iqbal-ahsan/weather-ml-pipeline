import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
DB_NAME = "ml_pipeline"
COLLECTION_NAME = "weather_raw"

def fetch_weather_data():
    print(f"[{datetime.now()}] fetching weather data from open-meteo")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 23.8103,
        "longitude": 90.4125,
        "start_date": "2025-04-01",
        "end_date": "2026-04-30",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "relative_humidity_2m_max"
        ],
        "timezone": "Asia/Dhaka"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data["daily"])
    df["fetched_at"] = datetime.now().isoformat()
    return df

def clean_data(df):
    print(f"[{datetime.now()}] cleaning data")
    df["time"] = pd.to_datetime(df["time"])

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)

    print(f"Removed {before - after} rows with missing values.")
    print(f"Clean data: {after} rows remaining ")
    return df

def save_to_mongodb(df):
    print("saving to Mongodb..")

    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    collection.drop()

    records = df.to_dict("records")
    collection.insert_many(records)

    print(f"saved {len(records)} records to MongoDB collection '{COLLECTION_NAME}'.")
    client.close()


def run_ingestion():
    print("=" * 50)
    print("Phase 2: Data Ingestion Started ")
    print("=" * 50)

    df = fetch_weather_data()
    df = clean_data(df)
    save_to_mongodb(df)

    print("=" * 50)
    print("Data Ingestion Completed ")
    print("=" * 50)
    return df

if __name__ == "__main__":
    run_ingestion()

