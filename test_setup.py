# test_setup.py (root folder এ বানাও)
from pymongo import MongoClient
import mlflow
import requests

print("Testing MongoDB...")
client = MongoClient("mongodb://localhost:27017/")
client.admin.command("ping")
print("✅ MongoDB connected")

print("Testing MLflow...")
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test")
with mlflow.start_run():
    mlflow.log_metric("test_metric", 1.0)
print("✅ MLflow connected")

print("Testing Open-Meteo API...")
r = requests.get("https://api.open-meteo.com/v1/forecast?latitude=23.8&longitude=90.4&current=temperature_2m")
print(f"✅ Weather API: {r.json()['current']['temperature_2m']}°C in Dhaka right now")

print("\n🎉 Phase 1 complete! সব ready.")