# 🌤️ Weather ML Pipeline — Dhaka Temperature Prediction

An end-to-end, production-grade machine learning pipeline that predicts daily maximum temperature in Dhaka, Bangladesh. Built with modern data engineering tools used in real-world ML teams.

---

## 📌 About

Most ML tutorials stop at model training. This project goes further — showing how a model actually gets from raw data into a scheduled, monitored, and deployable system.

Raw weather data is fetched daily from the Open-Meteo API, cleaned and transformed into meaningful features, then used to train and evaluate multiple ML models. Every experiment is tracked with MLflow, and the best-performing model is automatically registered and served as a live REST API via FastAPI. The entire pipeline runs on a daily schedule using Apache Airflow, with all services containerized in Docker.

---

## 🏗️ Architecture

```
Open-Meteo API
      │
      ▼
 [Ingest & Clean]  ──►  MongoDB (raw data)
      │
      ▼
 [Feature Engineering]  ──►  MongoDB (processed data)
      │
      ▼
 [Model Training]  ──►  MLflow (experiment tracking)
      │                       │
      │                       ▼
      │               Model Registry
      │                       │
      ▼                       ▼
 Apache Airflow        FastAPI (REST API)
 (orchestration)

All services containerized with Docker Compose
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data source | Open-Meteo API |
| Storage | MongoDB |
| Processing | Python, Pandas, Scikit-learn |
| Experiment tracking | MLflow |
| Orchestration | Apache Airflow |
| Serving | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |

---

## 📁 Project Structure

```
weather-ml-pipeline/
├── dags/
│   └── ml_pipeline_dag.py      # Airflow DAG
├── src/
│   ├── ingest.py               # Data fetching + MongoDB storage
│   ├── preprocess.py           # Feature engineering
│   ├── train.py                # Model training + MLflow tracking
│   └── predict.py              # Inference logic
├── api/
│   └── main.py                 # FastAPI app
├── data/
│   └── raw/
├── models/
├── docker-compose.yml
├── Dockerfile.airflow
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop
- Git

### 1. Clone the repository
```bash
git clone https://github.com/iqbal-ahsan/weather-ml-pipeline.git
cd weather-ml-pipeline
```

### 2. Start all services
```bash
docker compose up -d
```

Wait 60 seconds, then verify:
```bash
docker compose ps
```

### 3. Run the pipeline
```bash
python src/ingest.py        # Fetch and store data
python src/preprocess.py    # Feature engineering
python src/train.py         # Train and register model
```

### 4. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

---

## 🌐 Service URLs

| Service | URL |
|---|---|
| FastAPI (Swagger UI) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 |
| MongoDB UI | http://localhost:8081 |

Airflow credentials: `admin` / `admin`

---

## 📊 Features Engineered

| Feature | Description |
|---|---|
| `month`, `day_of_year`, `week`, `quarter` | Time-based features |
| `is_summer`, `is_monsoon`, `is_winter` | Seasonal flags |
| `temp_lag_1`, `temp_lag_3`, `temp_lag_7`, `temp_lag_14` | Lag features |
| `temp_rolling_7`, `temp_rolling_14`, `temp_rolling_30` | Rolling averages |
| `temp_std_7` | Rolling standard deviation |
| `precip_rolling_7`, `precip_rolling_30` | Precipitation rolling sums |

---

## 🤖 Models Trained

| Model | Description |
|---|---|
| Linear Regression | Baseline model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting model |

Best model is automatically selected by lowest MAE and registered in MLflow Model Registry.

---

## 🔌 API Usage

### Predict temperature

**POST** `/predict`

```json
{
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
```

**Response:**
```json
{
  "predicted_max_temperature": 35.27,
  "unit": "°C",
  "location": "Dhaka, Bangladesh"
}
```

---

## ⚙️ Airflow DAG

The pipeline runs automatically every day at midnight:

```
ingest_data → preprocess_data → train_model
```

- **Schedule**: `@daily`
- **Retries**: 2 (with 5 minute delay)
- **Owner**: iqbal-ahsan

---

## 📈 MLflow Experiment Tracking

Every training run logs:
- Model type and parameters
- MAE, RMSE, R² metrics
- Trained model artifact
- Best model registered in Model Registry

View all experiments at: **http://localhost:5000**

---

## 👤 Author

**Iqbal Ahsan**
- 📧 iqbalahsan4470@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/iqbal-ahsan)
