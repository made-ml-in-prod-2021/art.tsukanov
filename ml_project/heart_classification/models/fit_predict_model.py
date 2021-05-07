from typing import Union, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from heart_classification.entities import TrainingParams

SklearnClassifier = Union[LogisticRegression, SVC]


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_params: TrainingParams
) -> SklearnClassifier:
    if train_params.model_type == 'LogisticRegression':
        model = LogisticRegression(
            **train_params.model_params,
            random_state=train_params.random_state,
        )
    elif train_params.model_type == 'SVC':
        model = SVC(
            **train_params.model_params,
            random_state=train_params.random_state,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnClassifier,
    features: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray,
    target: pd.Series
) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(target, predicts),
    }
