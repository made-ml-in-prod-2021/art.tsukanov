from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.1)
    random_state: int = field(default=42)


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]
