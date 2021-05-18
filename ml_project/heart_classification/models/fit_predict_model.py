from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score


def train_model(
    model: ClassifierMixin,
    features: pd.DataFrame,
    target: pd.Series
) -> ClassifierMixin:
    if not hasattr(model, 'fit'):
        raise TypeError('Provided model is not a classifier instance. '
                        'Method "fit" must be implemented.')
    model.fit(features, target)
    return model


def predict_model(
    model: ClassifierMixin,
    features: pd.DataFrame
) -> np.ndarray:
    if not hasattr(model, 'predict'):
        raise TypeError('Provided model is not a classifier instance. '
                        'Method "predict" must be implemented.')
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray,
    target: pd.Series
) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(target, predicts),
    }
