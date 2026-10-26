"""
Automatic column data type and issue detection for tabular datasets.
"""

from typing import Dict, List, Set, Any
import pandas as pd
import numpy as np
from src.config import PreprocessingConfig


class ColumnDetector:
    """Detects semantic types and quality issues across dataset columns."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the DataFrame and categorizes columns into:
        - numeric
        - categorical
        - datetime
        - constant
        - high_missing
        - id_columns
        - target
        """
        num_rows = len(df)
        all_cols = list(df.columns)
        
        target = self.config.target_column
        feature_cols = [c for c in all_cols if c != target]

        constant_cols: List[str] = []
        high_missing_cols: List[str] = []
        id_cols: List[str] = []
        datetime_cols: List[str] = []
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []

        # Forced types
        forced_num = set(self.config.force_numeric_columns)
        forced_cat = set(self.config.force_categorical_columns)
        forced_dt = set(self.config.force_datetime_columns)
        forced_drop = set(self.config.columns_to_drop)

        for col in feature_cols:
            if col in forced_drop:
                continue

            series = df[col]
            missing_ratio = series.isna().mean()
            non_null = series.dropna()
            unique_count = non_null.nunique()

            # 1. High missing check
            if missing_ratio >= self.config.missing_threshold_drop:
                high_missing_cols.append(col)
                continue

            # 2. Constant column check
            if unique_count <= 1 and self.config.drop_constant_columns:
                constant_cols.append(col)
                continue

            # 3. Forced overrides
            if col in forced_dt:
                datetime_cols.append(col)
                continue
            if col in forced_num:
                numeric_cols.append(col)
                continue
            if col in forced_cat:
                categorical_cols.append(col)
                continue

            # 4. Check Numeric (direct or string-formatted currency/percentages)
            if pd.api.types.is_numeric_dtype(series) or self._can_be_numeric(non_null):
                # Check if it is an explicit numeric ID (like customer_id = 1001, 1002...)
                if self._is_explicit_id_name(col) and (unique_count / max(1, len(non_null))) > 0.9:
                    id_cols.append(col)
                else:
                    numeric_cols.append(col)
                continue

            # 5. Check Datetime
            if self._is_datetime(series):
                datetime_cols.append(col)
                continue

            # 6. Check string ID column
            if self._is_id_column(series, num_rows):
                id_cols.append(col)
                continue

            # 7. Boolean & Categorical
            categorical_cols.append(col)

        return {
            "all_columns": all_cols,
            "target": target,
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "constant": constant_cols,
            "high_missing": high_missing_cols,
            "id_columns": id_cols,
            "forced_drop": list(forced_drop),
        }

    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if a column is or can be parsed as datetime."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
            return False

        # Sample up to 50 rows for performance
        sample = non_null.astype(str).str.strip().head(50)
        
        # Avoid treating pure numbers (e.g., zip codes, years like 2024 without months) as datetime
        if sample.str.replace(r"[\.,\-]", "", regex=True).str.isnumeric().all() and sample.str.len().max() <= 5:
            return False

        try:
            # Test conversion
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            # If > 80% successfully parsed as dates
            return parsed.notna().mean() >= 0.8
        except Exception:
            return False

    def _is_explicit_id_name(self, col_name: str) -> bool:
        """Check if column name strongly hints at being an identifier."""
        name = str(col_name).lower()
        id_keywords = ["id", "uuid", "guid", "identifier", "ssn", "pk", "key"]
        return any(kw == name or name.endswith(f"_{kw}") or name.startswith(f"{kw}_") for kw in id_keywords)

    def _is_id_column(self, series: pd.Series, num_rows: int) -> bool:
        """Detect identifiers based on column name and high uniqueness."""
        col_name = str(series.name)
        non_null = series.dropna()
        unique_count = non_null.nunique()

        if len(non_null) == 0:
            return False

        cardinality_ratio = unique_count / len(non_null)

        # Check explicit naming clues
        if self._is_explicit_id_name(col_name) and cardinality_ratio > 0.7:
            return True

        # Pure string column with nearly all unique values
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            if cardinality_ratio >= self.config.id_cardinality_threshold and len(non_null) > 20:
                return True

        return False

    def _can_be_numeric(self, sample_series: pd.Series) -> bool:
        """Check if string series can be cleanly converted to numeric after removing currency/commas."""
        if len(sample_series) == 0:
            return False
        sample = sample_series.astype(str).str.strip().head(50)
        cleaned = sample.str.replace(r"[\$,% ]", "", regex=True)
        try:
            converted = pd.to_numeric(cleaned, errors="coerce")
            return converted.notna().mean() >= 0.85
        except Exception:
            return False
