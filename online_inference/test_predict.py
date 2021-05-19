import os

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score

from app import app

PATH_TO_DATASET = 'heart.csv'
TARGET_COLUMN = 'target'

os.environ['PATH_TO_MODEL'] = './model.pkl'


def test_predict_returns_400_on_wrong_number_of_features():
    df = pd.read_csv(PATH_TO_DATASET)
    data = df.drop(columns=TARGET_COLUMN).iloc[:, : len(df.columns) - 2]
    request_params = {
        'data': data.values.tolist(),
        'features': data.columns.tolist()
    }
    with TestClient(app) as client:
        response = client.get("/predict/", json=request_params)
        assert response.status_code == 400


def test_predict_returns_400_on_wrong_data_type():
    df = pd.read_csv(PATH_TO_DATASET)
    data = df.drop(columns=TARGET_COLUMN)
    data[0] = 'some string'
    print(data)
    request_params = {
        'data': data.values.tolist(),
        'features': data.columns.tolist()
    }
    with TestClient(app) as client:
        response = client.get("/predict/", json=request_params)
        assert response.status_code == 400


def test_predict_accuracy_is_ok():
    df = pd.read_csv(PATH_TO_DATASET)
    data = df.drop(columns=TARGET_COLUMN)
    target = df[TARGET_COLUMN]
    request_params = {
        'data': data.values.tolist(),
        'features': data.columns.tolist()
    }
    with TestClient(app) as client:
        response = client.get("/predict/", json=request_params)
        assert response.status_code == 200
        predictions = response.json()['disease']
        accuracy = accuracy_score(target, predictions)
        assert 0.8 < accuracy < 1.0
