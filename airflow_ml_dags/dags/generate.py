import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "generate_data",
        default_args=default_args,
        schedule_interval="@once",
        start_date=datetime.today(),
) as dag:
    generate = DockerOperator(
        image="generate",
        command="--output_dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="generate_data",
        do_xcom_push=False,
        volumes=["/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/data:/data"]
    )
