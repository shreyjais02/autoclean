"""
Configuration settings for the automated preprocessing pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """Configuration options for data preprocessing."""
    # Target column (excluded from feature scaling/transformations or handled separately)
    target_column: Optional[str] = None

    # Cleaning options
    drop_duplicates: bool = True
    drop_constant_columns: bool = True
    missing_threshold_drop: float = 0.80  # Drop column if > 80% values missing
    id_cardinality_threshold: float = 0.95 # Drop column if string/object and > 95% unique
    
    # Missing Value Imputation
    numeric_imputer_strategy: str = "median"  # 'mean', 'median', 'most_frequent', 'constant', 'drop'
    numeric_fill_value: Optional[float] = None
    categorical_imputer_strategy: str = "most_frequent"  # 'most_frequent', 'constant', 'drop'
    categorical_fill_value: str = "Missing"

    # Outlier Detection & Handling
    outlier_method: str = "iqr"  # 'iqr', 'zscore', 'none'
    outlier_action: str = "clip"  # 'clip', 'drop', 'none'
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0

    # Categorical Encoding
    encoding_strategy: str = "auto"  # 'auto', 'onehot', 'label', 'frequency', 'none'
    max_onehot_cardinality: int = 10  # If cardinality <= 10 use one-hot, else label/frequency

    # Numeric Scaling
    scaling_strategy: str = "standard"  # 'standard', 'minmax', 'robust', 'none'

    # Datetime handling
    extract_datetime_features: bool = True
    datetime_features: List[str] = field(default_factory=lambda: [
        "year", "month", "day", "dayofweek", "is_weekend", "hour"
    ])
    drop_original_datetime: bool = True

    # Custom column overrides (optional)
    force_numeric_columns: List[str] = field(default_factory=list)
    force_categorical_columns: List[str] = field(default_factory=list)
    force_datetime_columns: List[str] = field(default_factory=list)
    columns_to_drop: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        """Instantiate config from dictionary, ignoring unknown keys."""
        valid_keys = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PreprocessingConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
