import numpy as np
import pandas as pd

from src.entities import FeatureParams
from src.features import (
    build_transformer,
    make_features,
    extract_target,
)

DATASET_PATH = 'data/raw/heart.csv'


def test_can_make_features():
    df = pd.read_csv(DATASET_PATH)
    cat_features = []
    num_features = df.select_dtypes(include=np.number).columns.tolist()[:5]
    params = FeatureParams(
        categorical_features=cat_features,
        numerical_features=num_features,
        target_col='target'
    )
    transformer = build_transformer(params)
    transformer.fit(df)
    df_features = make_features(df, transformer)
    assert isinstance(df_features, pd.DataFrame)
    assert df_features.shape[0] == df.shape[0]
    assert df_features.shape[1] == len(num_features)


def test_can_extract_target():
    df = pd.read_csv(DATASET_PATH)
    params = FeatureParams(
        categorical_features=[],
        numerical_features=[],
        target_col='target'
    )
    target = extract_target(df, params)
    assert isinstance(target, pd.Series)
    assert target.shape[0] == df.shape[0]
