from math import floor, ceil

import pandas as pd

from heart_classification.entities import SplittingParams
from heart_classification.data import (
    read_data,
    split_data,
)

DATASET_PATH = 'data/raw/heart.csv'
VAL_SIZE = 0.1


def test_can_read_dataset():
    df = read_data(DATASET_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) >= 1
    assert len(df.index) >= 1


def test_can_split_dataset():
    df = pd.read_csv(DATASET_PATH)
    splitting_params = SplittingParams(val_size=VAL_SIZE)
    train_df, val_df = split_data(df, splitting_params)
    assert train_df.shape[0] == floor(df.shape[0] * (1 - VAL_SIZE)) or ceil(df.shape[0] * (1 - VAL_SIZE))
    assert train_df.shape[1] == df.shape[1]
    assert val_df.shape[0] == floor(df.shape[0] * VAL_SIZE) or ceil(df.shape[0] * VAL_SIZE)
    assert val_df.shape[1] == df.shape[1]
