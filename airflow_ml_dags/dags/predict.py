import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.models import Variable

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}
path_to_model = Variable.get("path_to_model")


with DAG(
        "make_predictions",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=datetime.today(),
) as dag:
    wait_for_data = FileSensor(
        filepath="./data/raw/{{ ds }}/",
        fs_conn_id="fs_default",
        mode="poke",
        poke_interval=10,
        timeout=600,
        retries=10,
        task_id="wait_for_data"
    )

    wait_for_model = FileSensor(
        filepath=f".{path_to_model}",
        fs_conn_id="fs_default",
        mode="poke",
        poke_interval=10,
        timeout=600,
        retries=10,
        task_id="wait_for_model"
    )

    predict = DockerOperator(
        image="predict",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/predictions/{{ ds }} " \
                f"--model-path {path_to_model}",
        network_mode="bridge",
        task_id="make_predictions",
        do_xcom_push=False,
        volumes=["/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/data:/data"]
    )

    [wait_for_data, wait_for_model] >> predict
