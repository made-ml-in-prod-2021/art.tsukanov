from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_features: List[str],
        numerical_features: List[str]
    ):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        self.is_fitted = True
        self.encoder.fit(X[self.categorical_features])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError(f'{self.__class__.__name__} is not fitted yet. Call "fit" '
                                 'first with appropriate arguments before using this transformer.')
        df_num = X[self.numerical_features].copy()
        df_num.fillna(df_num.mean(), inplace=True)

        df_categ = X[self.categorical_features].copy()
        df_categ.fillna(df_categ.mode().iloc[0], inplace=True)  # most frequent

        encoded_features = self.encoder.transform(df_categ)
        df_categ = pd.DataFrame(encoded_features, columns=df_categ.columns, index=df_categ.index)

        return pd.concat([df_num, df_categ], axis=1)
