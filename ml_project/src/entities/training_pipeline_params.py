from dataclasses import dataclass, field
from typing import List


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.1)
    random_state: int = field(default=42)


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str


@dataclass()
class TrainingParams:
    model_type: str = field(default='LogisticRegression')
    model_params: dict = field(default_factory=dict)
    random_state: int = field(default=42)


@dataclass()
class TrainingPipelineParams:
    logging_config_path: str
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    training_params: TrainingParams
