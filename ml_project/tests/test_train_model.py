import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import pytest
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

from heart_classification.models import (
    train_model,
    predict_model,
    evaluate_model,
)

DATASET_PATH = 'data/raw/heart.csv'
TARGET_COL = 'target'


@pytest.fixture
def features():
    df = pd.read_csv(DATASET_PATH)
    return df.drop(columns=TARGET_COL)


@pytest.fixture
def target():
    df = pd.read_csv(DATASET_PATH)
    return df[TARGET_COL]


def test_train_raises_error_on_no_fit_implemented(features, target):
    model = ClassifierMixin()
    with pytest.raises(TypeError):
        train_model(model, features, target)


def test_train_raises_error_on_no_predict_implemented(features):
    model = ClassifierMixin()
    with pytest.raises(TypeError):
        predict_model(model, features)


def get_model_predicts(features, target):
    model = LogisticRegression(max_iter=1000)
    train_model(model, features, target)
    predicts = predict_model(model, features)
    return predicts


def test_can_predict_model(features, target):
    predicts = get_model_predicts(features, target)
    assert isinstance(predicts, np.ndarray)
    assert predicts.shape == target.shape


def test_can_evaluate_model(features, target):
    predicts = get_model_predicts(features, target)
    metrics = evaluate_model(predicts, target)
    assert 'accuracy' in metrics
    assert 0.5 < metrics['accuracy'] < 1, f'accuracy is {metrics["accuracy"]}'


@pytest.fixture
def generated_data(features):
    size = np.random.randint(100, 500)
    generated_data = []
    for col in list(features.columns):
        generated_col = gaussian_kde(features[col]).resample(size=size, seed=42)[0]
        generated_data.append(generated_col)
    df_features = pd.DataFrame(np.array(generated_data).T, columns=features.columns)
    target = pd.Series(np.random.randint(2, size=size))
    return df_features, target


def test_can_process_generated_data(generated_data):
    gen_features, gen_target = generated_data
    predicts = get_model_predicts(gen_features, gen_target)
    metrics = evaluate_model(predicts, gen_target)
    assert 'accuracy' in metrics
    assert 0.5 < metrics['accuracy'] < 1, f'accuracy is {metrics["accuracy"]}'
