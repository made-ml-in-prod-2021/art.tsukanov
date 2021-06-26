import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.models import Variable

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "training_pipeline",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=datetime.today(),
) as dag:
    random_state = Variable.get("random_state")

    wait_for_data = FileSensor(
        filepath="./data/raw/{{ ds }}/",
        fs_conn_id="fs_default",
        mode="poke",
        poke_interval=10,
        timeout=600,
        retries=10,
        task_id="wait_for_data"
    )

    process_data = DockerOperator(
        image="train",
        command="process.py --input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        network_mode="bridge",
        task_id="process_data",
        do_xcom_push=False,
        volumes=["/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/data:/data"]
    )

    split_data = DockerOperator(
        image="train",
        command="split.py --input-dir /data/processed/{{ ds }} --output-dir /data/processed/{{ ds }} " \
                f"--random-state {random_state}",
        network_mode="bridge",
        task_id="split_data",
        do_xcom_push=False,
        volumes=["/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/data:/data"]
    )

    train_model = DockerOperator(
        image="train",
        command="train.py --input-dir /data/processed/{{ ds }} --models-dir /data/models/{{ ds }} " \
                f"--random-state {random_state}",
        task_id="train_model",
        do_xcom_push=False,
        volumes=["/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/data:/data"]
    )

    validate_model = DockerOperator(
        image="train",
        command="validate.py --input-dir /data/processed/{{ ds }} --models-dir /data/models/{{ ds }}",
        task_id="validate_model",
        do_xcom_push=False,
        volumes=["/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/data:/data"]
    )

    wait_for_data >> process_data >> split_data >> train_model >> validate_model
