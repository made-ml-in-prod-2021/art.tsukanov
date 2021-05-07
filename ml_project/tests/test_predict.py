import os.path
import json

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from heart_classification import predict
from heart_classification.entities import (
    FeatureParams,
    PredictionParams,
)

DATA_PATH = 'data/raw/heart.csv'
OUTPUT_PATH = 'tests/output.csv'


@pytest.fixture
def prediction_params():
    params = PredictionParams(
        logging_config_path='tests/logging_config.yaml',
        data_path=DATA_PATH,
        model_path='models/model_logreg.pkl',
        output_path=OUTPUT_PATH,
        feature_params=FeatureParams(
            categorical_features=[],
            numerical_features=[
                'age',
                'sex',
                'cp',
                'trestbps',
                'chol',
                'fbs',
                'restecg',
                'thalach',
                'exang',
                'oldpeak',
                'slope',
                'ca',
                'thal',
            ],
            target_col='target'
        ),
    )
    return params


def test_prediction(prediction_params):
    predict(prediction_params)
    assert os.path.isfile(OUTPUT_PATH)
    predicts = np.genfromtxt(OUTPUT_PATH, delimiter=',')
    input_dataset = pd.read_csv(DATA_PATH)
    target = input_dataset[prediction_params.feature_params.target_col]
    accuracy = accuracy_score(target, predicts)
    assert 0.8 < accuracy < 1, f'accuracy is {accuracy}'
