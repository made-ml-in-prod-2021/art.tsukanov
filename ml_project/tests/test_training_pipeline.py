import os.path
import json

import pytest

from src import training_pipeline
from src.entities import (
    SplittingParams,
    FeatureParams,
    TrainingParams,
    TrainingPipelineParams,
)

METRICS_PATH = 'tests/metrics.json'
MODEL_PATH = 'tests/model.pkl'


@pytest.fixture
def training_pipeline_params():
    params = TrainingPipelineParams(
        logging_config_path='tests/logging_config.yaml',
        input_data_path='data/raw/heart.csv',
        output_model_path=MODEL_PATH,
        metric_path=METRICS_PATH,
        splitting_params=SplittingParams(
            val_size=0.1,
            random_state=42,
        ),
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
            target_col='target',
        ),
        training_params=TrainingParams(
            model_type='LogisticRegression',
            model_params={
                'C': 0.1,
                'max_iter': 1000,
            },
            random_state=42,
        ),
    )
    return params


def test_pipeline(training_pipeline_params):
    training_pipeline(training_pipeline_params)
    assert os.path.isfile(MODEL_PATH)
    assert os.path.isfile(METRICS_PATH)

    with open(METRICS_PATH, 'r') as fin:
        metrics = json.load(fin)
    assert 'accuracy' in metrics
    assert 0.5 < metrics['accuracy'] < 1, f'accuracy is {metrics["accuracy"]}'
