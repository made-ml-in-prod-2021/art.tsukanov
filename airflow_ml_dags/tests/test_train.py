import os
import sys
sys.path.append("./images/train")
import json

from airflow.models import DagBag
from unittest.mock import patch
from sklearn.datasets import load_iris

from process import process_data
from split import split_data
from train import train_model
from validate import validate_model

DAG_PATH = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/dags/train.py"
INPUT_DIR = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/tests/test_data/train/raw"
OUTPUT_DIR = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/tests/test_data/train/processed"
MODELS_DIR = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/tests/test_data/train/models"
RANDOM_STATE = 42


@patch('airflow.hooks.base.BaseHook.get_connection')
@patch('airflow.models.Variable')
def test_dag_load_with_no_errors(_, __):
    dag_bag = DagBag(include_examples=False)
    dag_bag.process_file(DAG_PATH)
    assert len(dag_bag.import_errors) == 0
    assert len(dag_bag.dags) == 1
    assert len(dag_bag.dags['training_pipeline'].tasks) == 5


def test_train_pipeline():
    X, y = load_iris(return_X_y=True, as_frame=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    X.to_csv(os.path.join(INPUT_DIR, "data.csv"), index=False)
    y.to_csv(os.path.join(INPUT_DIR, "target.csv"), index=False)

    process_data(INPUT_DIR, OUTPUT_DIR)
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "data.csv"))
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "target.csv"))

    split_data(OUTPUT_DIR, OUTPUT_DIR, RANDOM_STATE)
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "X_train.csv"))
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "y_train.csv"))
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "X_val.csv"))
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "y_val.csv"))

    train_model(OUTPUT_DIR, MODELS_DIR, RANDOM_STATE)
    assert os.path.isfile(os.path.join(MODELS_DIR, "logreg.pth"))

    validate_model(OUTPUT_DIR, MODELS_DIR)
    assert os.path.isfile(os.path.join(MODELS_DIR, "metrics.json"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as fin:
        metrics = json.load(fin)
    assert metrics["accuracy"] > 0.8
