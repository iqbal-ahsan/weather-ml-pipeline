import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "ml_pipeline"
RAW_COLLECTION = "weather_raw"
PROCESSED_COLLECTION = "weather_processed"

def load_from_mongodb():
    print("Loading raw data from MongoDB...")

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    cursor = db[RAW_COLLECTION].find({}, {"_id": 0, "fetched_at": 0})
    df = pd.DataFrame(list(cursor))
    client.close()

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    print(f"Loaded {len(df)} rows from MongoDB.")
    return df

def create_time_features(df):
    print("Creating time-based features...")

    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    df["week"] = df["time"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["time"].dt.quarter
    df["is_summer"] = df["month"].isin([3, 4, 5]).astype(int)
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["is_winter"] = df["month"].isin([11, 12, 1, 2]).astype(int)

    return df

def create_lag_features(df):
    print("Creating lag features...")

    df["temp_lag_1"] = df["temperature_2m_max"].shift(1)
    df["temp_lag_3"] = df["temperature_2m_max"].shift(3)
    df["temp_lag_7"] = df["temperature_2m_max"].shift(7)
    df["temp_lag_14"] = df["temperature_2m_max"].shift(14)

    df["precip_lag_1"] = df["precipitation_sum"].shift(1)
    df["precip_lag_7"] = df["precipitation_sum"].shift(7)

    return df

def create_rolling_features(df):
    print("Creating rolling average features...")

    df["temp_rolling_7"] = df["temperature_2m_max"].rolling(window=7).mean()
    df["temp_rolling_14"] = df["temperature_2m_max"].rolling(window=14).mean()
    df["temp_rolling_30"] = df["temperature_2m_max"].rolling(window=30).mean()

    df["temp_std_7"] = df["temperature_2m_max"].rolling(window=7).std()

    df["precip_rolling_7"] = df["precipitation_sum"].rolling(window=7).sum()
    df["precip_rolling_30"] = df["precipitation_sum"].rolling(window=30).sum()

    return df

def save_processed_data(df):
    print("Saving processed data to MongoDB...")

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[PROCESSED_COLLECTION]

    collection.drop()

    df["time"] = df["time"].astype(str)
    records = df.to_dict("records")
    collection.insert_many(records)

    print(f"Saved {len(records)} processed records to '{PROCESSED_COLLECTION}'.")
    client.close()

def run_preprocessing():
    print("=" * 50)
    print("PHASE 3: FEATURE ENGINEERING STARTED")
    print("=" * 50)

    df = load_from_mongodb()
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    after = len(df)
    print(f"Dropped {before - after} rows due to lag/rolling NaN values.")

    save_processed_data(df)

    print("\nFeature Summary:")
    print(f"  Total features : {len(df.columns)}")
    print(f"  Total rows     : {len(df)}")
    print(f"  Date range     : {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print("=" * 50)
    print("FEATURE ENGINEERING COMPLETE ✅")
    print("=" * 50)

    return df

if __name__ == "__main__":
    run_preprocessing()