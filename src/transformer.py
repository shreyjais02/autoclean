"""
Feature transformation: Datetime extraction, Categorical Encoding, and Numeric Scaling.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.config import PreprocessingConfig


class FeatureTransformer:
    """Transforms features through datetime extraction, categorical encoding, and scaling."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scaler: Optional[Any] = None
        self.scaled_columns_: List[str] = []
        self.onehot_columns_: List[str] = []
        self.onehot_categories_: Dict[str, List[Any]] = {}
        self.label_encoders_: Dict[str, Dict[Any, int]] = {}
        self.freq_encoders_: Dict[str, Dict[Any, float]] = {}
        self.datetime_columns_: List[str] = []
        self.extracted_datetime_cols_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, detected: Dict[str, Any]) -> "FeatureTransformer":
        """Fit encoding and scaling on DataFrame."""
        df_work = df.copy()

        # 1. Identify active datetime columns
        self.datetime_columns_ = [c for c in detected["datetime"] if c in df_work.columns]

        # 2. Simulate datetime transformation for proper schema discovery
        if self.config.extract_datetime_features:
            df_work = self._extract_datetime_features(df_work)

        # 3. Categorical encoding fit
        categorical_cols = [
            c for c in detected["categorical"] 
            if c in df_work.columns and c != self.config.target_column
        ]

        strategy = self.config.encoding_strategy
        self.onehot_columns_ = []
        self.label_encoders_ = {}
        self.freq_encoders_ = {}

        for col in categorical_cols:
            series = df_work[col].astype(str)
            unique_vals = list(series.unique())
            cardinality = len(unique_vals)

            use_onehot = (
                strategy == "onehot" or
                (strategy == "auto" and cardinality <= self.config.max_onehot_cardinality)
            )

            if use_onehot and strategy != "none":
                self.onehot_columns_.append(col)
                self.onehot_categories_[col] = unique_vals
            elif strategy in ("label", "auto") and strategy != "none":
                # Label encoding mapping with fallback for unseen
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.label_encoders_[col] = mapping
            elif strategy == "frequency" and strategy != "none":
                freq_map = (series.value_counts(normalize=True)).to_dict()
                self.freq_encoders_[col] = freq_map

        # Apply simulated encoding to determine numeric columns for scaler
        df_encoded = self._encode_categoricals(df_work)

        # 4. Numeric scaling fit
        if self.config.scaling_strategy != "none":
            # Select all numeric columns except target
            numeric_cols = [
                c for c in df_encoded.columns
                if c != self.config.target_column and pd.api.types.is_numeric_dtype(df_encoded[c])
            ]
            self.scaled_columns_ = numeric_cols

            if self.scaled_columns_:
                if self.config.scaling_strategy == "minmax":
                    self.scaler = MinMaxScaler()
                elif self.config.scaling_strategy == "robust":
                    self.scaler = RobustScaler()
                else:  # default: standard
                    self.scaler = StandardScaler()

                self.scaler.fit(df_encoded[self.scaled_columns_])

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply datetime extraction, encoding, and scaling to DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("FeatureTransformer must be fitted before calling transform.")

        df_trans = df.copy()

        # 1. Datetime feature extraction
        if self.config.extract_datetime_features and self.datetime_columns_:
            df_trans = self._extract_datetime_features(df_trans)

        # 2. Categorical encoding
        if self.config.encoding_strategy != "none":
            df_trans = self._encode_categoricals(df_trans)

        # 3. Numeric scaling
        if self.scaler is not None and self.scaled_columns_:
            existing_scale_cols = [c for c in self.scaled_columns_ if c in df_trans.columns]
            if existing_scale_cols:
                scaled_values = self.scaler.transform(df_trans[existing_scale_cols])
                df_trans[existing_scale_cols] = scaled_values

        return df_trans

    def fit_transform(self, df: pd.DataFrame, detected: Dict[str, Any]) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, detected).transform(df)

    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract calendar & time features from detected datetime columns."""
        df_out = df.copy()
        new_cols = []
        for col in self.datetime_columns_:
            if col in df_out.columns:
                parsed = pd.to_datetime(df_out[col], errors="coerce", format="mixed")
                
                req_feats = self.config.datetime_features
                if "year" in req_feats:
                    name = f"{col}_year"
                    df_out[name] = parsed.dt.year.fillna(-1).astype(int)
                    new_cols.append(name)
                if "month" in req_feats:
                    name = f"{col}_month"
                    df_out[name] = parsed.dt.month.fillna(-1).astype(int)
                    new_cols.append(name)
                if "day" in req_feats:
                    name = f"{col}_day"
                    df_out[name] = parsed.dt.day.fillna(-1).astype(int)
                    new_cols.append(name)
                if "dayofweek" in req_feats or "day_of_week" in req_feats:
                    name = f"{col}_dayofweek"
                    dow = parsed.dt.day_of_week if hasattr(parsed.dt, "day_of_week") else parsed.dt.dayofweek
                    df_out[name] = dow.fillna(-1).astype(int)
                    new_cols.append(name)
                if "is_weekend" in req_feats:
                    name = f"{col}_is_weekend"
                    dow = parsed.dt.day_of_week if hasattr(parsed.dt, "day_of_week") else parsed.dt.dayofweek
                    df_out[name] = dow.isin([5, 6]).astype(int)
                    new_cols.append(name)
                if "hour" in req_feats:
                    name = f"{col}_hour"
                    df_out[name] = parsed.dt.hour.fillna(-1).astype(int)
                    new_cols.append(name)

                if self.config.drop_original_datetime:
                    df_out = df_out.drop(columns=[col])

        self.extracted_datetime_cols_ = new_cols
        return df_out

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply One-Hot, Label, or Frequency encoding."""
        df_out = df.copy()

        # 1. One-Hot Encoding
        for col in self.onehot_columns_:
            if col in df_out.columns:
                categories = self.onehot_categories_.get(col, [])
                series = df_out[col].astype(str)
                for cat in categories:
                    dummy_col_name = f"{col}_{cat}"
                    df_out[dummy_col_name] = (series == cat).astype(int)
                df_out = df_out.drop(columns=[col])

        # 2. Label Encoding
        for col, mapping in self.label_encoders_.items():
            if col in df_out.columns:
                series = df_out[col].astype(str)
                # Unseen categories get mapped to -1
                df_out[col] = series.map(mapping).fillna(-1).astype(int)

        # 3. Frequency Encoding
        for col, freq_map in self.freq_encoders_.items():
            if col in df_out.columns:
                series = df_out[col].astype(str)
                df_out[col] = series.map(freq_map).fillna(0.0).astype(float)

        return df_out
