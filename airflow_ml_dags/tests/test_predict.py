import os
import sys
sys.path.append("./images/predict")
import json

from airflow.models import DagBag
from unittest.mock import patch
from sklearn.datasets import load_iris

from predict import make_predictions

DAG_PATH = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/dags/predict.py"
INPUT_DIR = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/tests/test_data/predict/raw"
OUTPUT_DIR = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/tests/test_data/predict/predictions"
MODEL_PATH = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/tests/model.pth"


@patch('airflow.hooks.base.BaseHook.get_connection')
@patch('airflow.models.Variable')
def test_dag_load_with_no_errors(_, __):
    dag_bag = DagBag(include_examples=False)
    dag_bag.process_file(DAG_PATH)
    assert len(dag_bag.import_errors) == 0
    assert len(dag_bag.import_errors) == 0
    assert len(dag_bag.dags) == 1
    assert len(dag_bag.dags['make_predictions'].tasks) == 3


def test_predicted_data_exist():
    X, _ = load_iris(return_X_y=True, as_frame=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    X.to_csv(os.path.join(INPUT_DIR, "data.csv"), index=False)

    make_predictions(INPUT_DIR, OUTPUT_DIR, MODEL_PATH)
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "predictions.csv"))
