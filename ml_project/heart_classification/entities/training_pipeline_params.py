from dataclasses import dataclass

from .params import (
    SplittingParams,
    FeatureParams,
    TrainingParams
)


@dataclass()
class TrainingPipelineParams:
    logging_config_path: str
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    training_params: TrainingParams
