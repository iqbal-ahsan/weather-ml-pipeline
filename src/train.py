import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DB_NAME = "ml_pipeline"
PROCESSED_COLLECTION = "weather_processed"

FEATURES = [
    "month", "day_of_year", "week", "quarter",
    "is_summer", "is_monsoon", "is_winter",
    "temp_lag_1", "temp_lag_3", "temp_lag_7", "temp_lag_14",
    "precip_lag_1", "precip_lag_7",
    "temp_rolling_7", "temp_rolling_14", "temp_rolling_30",
    "temp_std_7", "precip_rolling_7", "precip_rolling_30",
    "temperature_2m_min", "precipitation_sum", "windspeed_10m_max",
    "relative_humidity_2m_max"
]
TARGET = "temperature_2m_max"

def load_processed_data():
    print("Loading processed data from MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    cursor = db[PROCESSED_COLLECTION].find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    client.close()
    print(f"Loaded {len(df)} rows.")
    return df

def get_X_y(df):
    available = [f for f in FEATURES if f in df.columns]
    X = df[available]
    y = df[TARGET]
    return X, y

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2

def train_model():
    print("=" * 50)
    print("PHASE 4: MODEL TRAINING STARTED")
    print("=" * 50)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("weather-temperature-prediction")

    df = load_processed_data()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
    }

    best_mae = float("inf")
    best_run_id = None
    best_model_name = None

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}...")

        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            mae, rmse, r2 = evaluate(model, X_test, y_test)

            # MLflow এ log করো
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"  MAE  : {mae:.3f}°C")
            print(f"  RMSE : {rmse:.3f}°C")
            print(f"  R2   : {r2:.3f}")

            run_id = mlflow.active_run().info.run_id

            if mae < best_mae:
                best_mae = mae
                best_run_id = run_id
                best_model_name = model_name


    print(f"\nBest model: {best_model_name} (MAE: {best_mae:.3f}°C)")
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, "WeatherTempModel")
    print("Best model registered in MLflow Model Registry ✅")

    print("=" * 50)
    print("MODEL TRAINING COMPLETE ✅")
    print("=" * 50)

if __name__ == "__main__":
    train_model()