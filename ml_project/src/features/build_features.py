import pandas as pd

from src.entities import (
    FeatureParams,
    FeatureTransformer,
)


def build_transformer(params: FeatureParams) -> FeatureTransformer:
    transformer = FeatureTransformer(
        params.categorical_features,
        params.numerical_features
    )
    return transformer


def make_features(df: pd.DataFrame, transformer: FeatureTransformer) -> pd.DataFrame:
    features = transformer.transform(df)
    return features


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
