from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI(
    title="Weather Temperature Prediction API",
    description="Predicts daily max temperature in Dhaka using ML",
    version="1.0.0"
)

# Model load করো
try:
    model = mlflow.sklearn.load_model("models:/WeatherTempModel/1")
    print("✅ Model loaded successfully from MLflow")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

class WeatherInput(BaseModel):
    month: int
    day_of_year: int
    week: int
    quarter: int
    is_summer: int
    is_monsoon: int
    is_winter: int
    temp_lag_1: float
    temp_lag_3: float
    temp_lag_7: float
    temp_lag_14: float
    precip_lag_1: float
    precip_lag_7: float
    temp_rolling_7: float
    temp_rolling_14: float
    temp_rolling_30: float
    temp_std_7: float
    precip_rolling_7: float
    precip_rolling_30: float
    temperature_2m_min: float
    precipitation_sum: float
    windspeed_10m_max: float
    relative_humidity_2m_max: float

@app.get("/")
def root():
    return {
        "message": "Weather Prediction API is running ✅",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: WeatherInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]

    return {
        "predicted_max_temperature": round(float(prediction), 2),
        "unit": "°C",
        "location": "Dhaka, Bangladesh"
    }

@app.get("/sample-input")
def sample_input():
    return {
        "month": 4,
        "day_of_year": 100,
        "week": 15,
        "quarter": 2,
        "is_summer": 1,
        "is_monsoon": 0,
        "is_winter": 0,
        "temp_lag_1": 35.2,
        "temp_lag_3": 34.8,
        "temp_lag_7": 33.9,
        "temp_lag_14": 32.5,
        "precip_lag_1": 0.0,
        "precip_lag_7": 2.3,
        "temp_rolling_7": 34.5,
        "temp_rolling_14": 33.8,
        "temp_rolling_30": 32.1,
        "temp_std_7": 1.2,
        "precip_rolling_7": 5.4,
        "precip_rolling_30": 18.2,
        "temperature_2m_min": 26.3,
        "precipitation_sum": 0.0,
        "windspeed_10m_max": 12.4,
        "relative_humidity_2m_max": 78.0
    }
