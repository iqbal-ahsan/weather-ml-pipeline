from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, "/opt/airflow/src")

default_args = {
    "owner": "iqbal-ahsan",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="weather_ml_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline for Dhaka weather prediction",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "weather", "dhaka"],
) as dag:

    def ingest_task():
        from ingest import run_ingestion
        run_ingestion()

    def preprocess_task():
        from preprocess import run_preprocessing
        run_preprocessing()

    def train_task():
        from train import train_model
        train_model()

    t1 = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_task,
    )

    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_task,
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=train_task,
    )

    t1 >> t2 >> t3