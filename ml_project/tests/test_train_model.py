import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from heart_classification.entities import TrainingParams
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


@pytest.fixture
def training_params_logreg():
    params = TrainingParams(
        model_type='LogisticRegression',
        model_params={
            'C': 0.1,
            'max_iter': 1000
        },
        random_state=42
    )
    return params


def test_can_train_logistic_regression(features, target, training_params_logreg):
    model = train_model(features, target, training_params_logreg)
    assert isinstance(model, LogisticRegression)


def test_can_train_svc(features, target):
    params = TrainingParams(
        model_type='SVC',
        model_params={
            'C': 1000
        },
        random_state=42
    )
    model = train_model(features, target, params)
    assert isinstance(model, SVC)


def test_train_raises_error_on_model_type_not_found(features, target):
    params = TrainingParams(
        model_type='UnknownModel'
    )
    with pytest.raises(NotImplementedError):
        train_model(features, target, params)


def get_model_predicts(features, target, training_params_logreg):
    model = train_model(features, target, training_params_logreg)
    predicts = predict_model(model, features)
    return predicts


def test_can_predict_model(features, target, training_params_logreg):
    predicts = get_model_predicts(features, target, training_params_logreg)
    assert isinstance(predicts, np.ndarray)
    assert predicts.shape == target.shape


def test_can_evaluate_model(features, target, training_params_logreg):
    predicts = get_model_predicts(features, target, training_params_logreg)
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


def test_can_process_generated_data(generated_data, training_params_logreg):
    gen_features, gen_target = generated_data
    predicts = get_model_predicts(gen_features, gen_target, training_params_logreg)
    metrics = evaluate_model(predicts, gen_target)
    assert 'accuracy' in metrics
    assert 0.5 < metrics['accuracy'] < 1, f'accuracy is {metrics["accuracy"]}'
