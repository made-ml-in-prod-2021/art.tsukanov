import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from heart_classification.entities import FeatureTransformer

DATASET_PATH = 'data/raw/heart.csv'


def test_raises_error_on_transform_before_fit():
    df = pd.DataFrame()
    transformer = FeatureTransformer([], [])
    with pytest.raises(NotFittedError):
        transformer.transform(df)


def test_can_transform_features():
    df = pd.read_csv(DATASET_PATH)
    cat_features = []
    num_features = df.select_dtypes(include=np.number).columns.tolist()[:5]
    transformer = FeatureTransformer(cat_features, num_features)
    transformer.fit(df)
    transformed_df = transformer.transform(df)
    assert transformed_df.shape[0] == df.shape[0]
    assert transformed_df.shape[1] == len(num_features)
