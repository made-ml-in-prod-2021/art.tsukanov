from dataclasses import dataclass

from .params import FeatureParams


@dataclass()
class PredictionParams:
    logging_config_path: str
    data_path: str
    model_path: str
    output_path: str
    feature_params: FeatureParams
