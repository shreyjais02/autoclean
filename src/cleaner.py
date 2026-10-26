"""
Data cleaning operations: deduplication, column pruning, numeric normalization,
missing value imputation, and outlier treatment.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.config import PreprocessingConfig


class DataCleaner:
    """Handles missing data, duplicates, outliers, and column pruning."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.dropped_columns_: List[str] = []
        self.duplicates_removed_: int = 0
        self.numeric_impute_values_: Dict[str, float] = {}
        self.categorical_impute_values_: Dict[str, Any] = {}
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.outliers_clipped_count_: Dict[str, int] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, detected: Dict[str, Any]) -> "DataCleaner":
        """Compute statistics for cleaning on training data."""
        # 1. Determine columns to drop
        self.dropped_columns_ = list(set(
            detected["constant"] +
            detected["high_missing"] +
            detected["id_columns"] +
            detected["forced_drop"]
        ))

        # Filter working dataframe
        working_cols = [c for c in df.columns if c not in self.dropped_columns_]
        df_work = df[working_cols].copy()

        # Clean numeric-like strings first if needed
        for col in detected["numeric"]:
            if col in df_work.columns:
                df_work[col] = self._clean_numeric_series(df_work[col])

        # 2. Compute numeric imputation values
        for col in detected["numeric"]:
            if col in df_work.columns:
                series = df_work[col].dropna()
                if len(series) > 0:
                    if self.config.numeric_imputer_strategy == "mean":
                        val = float(series.mean())
                    elif self.config.numeric_imputer_strategy == "most_frequent":
                        mode_vals = series.mode()
                        val = float(mode_vals[0]) if len(mode_vals) > 0 else 0.0
                    elif self.config.numeric_imputer_strategy == "constant":
                        val = float(self.config.numeric_fill_value if self.config.numeric_fill_value is not None else 0.0)
                    else:  # default: median
                        val = float(series.median())
                else:
                    val = 0.0
                self.numeric_impute_values_[col] = val

        # 3. Compute categorical imputation values
        for col in detected["categorical"]:
            if col in df_work.columns:
                series = df_work[col].dropna()
                if self.config.categorical_imputer_strategy == "constant":
                    val = self.config.categorical_fill_value
                else:  # default: most_frequent
                    if len(series) > 0:
                        mode_vals = series.mode()
                        val = mode_vals[0] if len(mode_vals) > 0 else self.config.categorical_fill_value
                    else:
                        val = self.config.categorical_fill_value
                self.categorical_impute_values_[col] = val

        # 4. Compute outlier boundaries
        if self.config.outlier_method in ("iqr", "zscore") and self.config.outlier_action != "none":
            for col in detected["numeric"]:
                if col in df_work.columns:
                    series = df_work[col].fillna(self.numeric_impute_values_.get(col, 0.0))
                    if len(series) > 0:
                        if self.config.outlier_method == "iqr":
                            q1 = float(series.quantile(0.25))
                            q3 = float(series.quantile(0.75))
                            iqr = q3 - q1
                            k = self.config.iqr_multiplier
                            lower = q1 - k * iqr
                            upper = q3 + k * iqr
                        else:  # zscore
                            mean = float(series.mean())
                            std = float(series.std(ddof=0))
                            if std == 0:
                                lower, upper = mean, mean
                            else:
                                z = self.config.zscore_threshold
                                lower = mean - z * std
                                upper = mean + z * std
                        self.outlier_bounds_[col] = (lower, upper)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, detected: Dict[str, Any]) -> pd.DataFrame:
        """Apply fitted cleaning rules to DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("DataCleaner must be fitted before calling transform.")

        df_clean = df.copy()

        # 1. Prune dropped columns (constant, high missing, id, forced drops)
        cols_to_remove = [c for c in self.dropped_columns_ if c in df_clean.columns]
        if cols_to_remove:
            df_clean = df_clean.drop(columns=cols_to_remove)

        # 2. Deduplication across meaningful columns
        if self.config.drop_duplicates:
            initial_len = len(df_clean)
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            self.duplicates_removed_ = initial_len - len(df_clean)

        # 3. Clean numeric columns & impute missing
        for col in detected["numeric"]:
            if col in df_clean.columns:
                df_clean[col] = self._clean_numeric_series(df_clean[col])
                fill_val = self.numeric_impute_values_.get(col, 0.0)
                df_clean[col] = df_clean[col].fillna(fill_val)

        # 4. Impute categorical missing
        for col in detected["categorical"]:
            if col in df_clean.columns:
                fill_val = self.categorical_impute_values_.get(col, self.config.categorical_fill_value)
                # Ensure string type if filling with string
                df_clean[col] = df_clean[col].fillna(fill_val)
                # Strip leading/trailing whitespaces if strings
                if df_clean[col].dtype == object or pd.api.types.is_string_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].astype(str).str.strip()

        # 5. Outlier handling
        self.outliers_clipped_count_ = {}
        if self.config.outlier_action == "clip":
            for col, (lower, upper) in self.outlier_bounds_.items():
                if col in df_clean.columns:
                    mask = (df_clean[col] < lower) | (df_clean[col] > upper)
                    clipped_count = int(mask.sum())
                    self.outliers_clipped_count_[col] = clipped_count
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        elif self.config.outlier_action == "drop":
            for col, (lower, upper) in self.outlier_bounds_.items():
                if col in df_clean.columns:
                    mask = (df_clean[col] >= lower) & (df_clean[col] <= upper)
                    df_clean = df_clean[mask].reset_index(drop=True)

        return df_clean

    def fit_transform(self, df: pd.DataFrame, detected: Dict[str, Any]) -> pd.DataFrame:
        """Fit and clean DataFrame in one step."""
        return self.fit(df, detected).transform(df, detected)

    @staticmethod
    def _clean_numeric_series(series: pd.Series) -> pd.Series:
        """Convert string series with symbols/commas to float."""
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        # Clean currency, commas, percentages, spaces
        cleaned = series.astype(str).str.replace(r"[\$,% ]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")
