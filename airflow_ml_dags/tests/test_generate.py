import os
import sys
sys.path.append("./images/generate")

from airflow.models import DagBag
from unittest.mock import patch

from generate import generate_data

DAG_PATH = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/dags/generate.py"
OUTPUT_DIR = "/Users/artem/tmp/made/ml_in_production/art.tsukanov/airflow_ml_dags/tests/test_data/generate/raw"


@patch('airflow.hooks.base.BaseHook.get_connection')
def test_dag_load_with_no_errors(_):
    dag_bag = DagBag(include_examples=False)
    dag_bag.process_file(DAG_PATH)
    assert len(dag_bag.import_errors) == 0
    assert len(dag_bag.import_errors) == 0
    assert len(dag_bag.dags) == 1
    assert len(dag_bag.dags['generate_data'].tasks) == 1


def test_generated_data_exist():
    generate_data(OUTPUT_DIR)
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "data.csv"))
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "target.csv"))
