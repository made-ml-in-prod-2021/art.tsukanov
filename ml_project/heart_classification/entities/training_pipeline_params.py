from dataclasses import dataclass, field

from .params import (
    SplittingParams,
    FeatureParams,
)


@dataclass()
class TrainingPipelineParams:
    logging_config_path: str
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    model: dict = field(default_factory={
        '_target_': 'sklearn.linear_model.LogisticRegression',
        'C': 0.1,
        'max_iter': 1000,
        'random_state': 42,
    })
