"""
Preprocessing reporting and audit generation.
"""

from typing import Dict, Any, List
import json
from pathlib import Path
import pandas as pd


class PreprocessingReporter:
    """Collects statistics during preprocessing and generates audit reports."""

    def __init__(self):
        self.stats: Dict[str, Any] = {
            "input_rows": 0,
            "input_columns": 0,
            "output_rows": 0,
            "output_columns": 0,
            "duplicates_removed": 0,
            "dropped_columns": {},
            "detected_types": {},
            "imputed_missing_values": {},
            "outliers_treated": {},
            "encoded_columns": {},
            "scaled_columns": [],
            "extracted_datetime_features": [],
            "execution_time_seconds": 0.0,
        }

    def record_input(self, df: pd.DataFrame) -> None:
        self.stats["input_rows"] = int(df.shape[0])
        self.stats["input_columns"] = int(df.shape[1])
        self.stats["input_memory_mb"] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    def record_output(self, df: pd.DataFrame, exec_time: float) -> None:
        self.stats["output_rows"] = int(df.shape[0])
        self.stats["output_columns"] = int(df.shape[1])
        self.stats["output_memory_mb"] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        self.stats["execution_time_seconds"] = round(exec_time, 3)

    def record_detections(self, detected: Dict[str, Any]) -> None:
        self.stats["detected_types"] = {
            "numeric": detected["numeric"],
            "categorical": detected["categorical"],
            "datetime": detected["datetime"],
            "constant": detected["constant"],
            "high_missing": detected["high_missing"],
            "id_columns": detected["id_columns"],
        }
        self.stats["dropped_columns"] = {
            "constant_columns": detected["constant"],
            "high_missing_columns": detected["high_missing"],
            "id_columns": detected["id_columns"],
            "user_forced_drop": detected["forced_drop"],
        }

    def record_cleaning(self, cleaner) -> None:
        self.stats["duplicates_removed"] = cleaner.duplicates_removed_
        self.stats["imputed_missing_values"] = {
            "numeric_imputations": cleaner.numeric_impute_values_,
            "categorical_imputations": cleaner.categorical_impute_values_,
        }
        self.stats["outliers_treated"] = cleaner.outliers_clipped_count_

    def record_transformations(self, transformer) -> None:
        self.stats["encoded_columns"] = {
            "one_hot": transformer.onehot_columns_,
            "label_encoded": list(transformer.label_encoders_.keys()),
            "frequency_encoded": list(transformer.freq_encoders_.keys()),
        }
        self.stats["scaled_columns"] = transformer.scaled_columns_
        self.stats["extracted_datetime_features"] = transformer.extracted_datetime_cols_

    def generate_text_summary(self) -> str:
        """Create a human-readable text summary of the preprocessing."""
        lines = [
            "=" * 60,
            "              DATA PREPROCESSING AUDIT REPORT              ",
            "=" * 60,
            f" Input Dataset Shape:      {self.stats['input_rows']} rows x {self.stats['input_columns']} columns ({self.stats.get('input_memory_mb', 0)} MB)",
            f" Output Dataset Shape:     {self.stats['output_rows']} rows x {self.stats['output_columns']} columns ({self.stats.get('output_memory_mb', 0)} MB)",
            f" Execution Time:           {self.stats['execution_time_seconds']} seconds",
            "-" * 60,
            "1. CLEANING ACTIONS:",
            f"   - Duplicates removed:   {self.stats['duplicates_removed']}",
        ]

        dropped = self.stats["dropped_columns"]
        all_dropped = sum(len(v) for v in dropped.values())
        lines.append(f"   - Columns dropped ({all_dropped}):")
        if dropped.get("constant_columns"):
            lines.append(f"     * Constant columns: {', '.join(dropped['constant_columns'])}")
        if dropped.get("high_missing_columns"):
            lines.append(f"     * High-missing columns: {', '.join(dropped['high_missing_columns'])}")
        if dropped.get("id_columns"):
            lines.append(f"     * ID / High-uniqueness columns: {', '.join(dropped['id_columns'])}")
        if dropped.get("user_forced_drop"):
            lines.append(f"     * User-requested drop: {', '.join(dropped['user_forced_drop'])}")
        if all_dropped == 0:
            lines.append("     * None")

        outliers = self.stats["outliers_treated"]
        total_outliers = sum(outliers.values()) if outliers else 0
        lines.append(f"   - Outlier values clipped: {total_outliers}")
        for col, count in outliers.items():
            if count > 0:
                lines.append(f"     * {col}: {count} values clipped")

        lines.extend([
            "-" * 60,
            "2. FEATURE TRANSFORMATIONS:",
            f"   - Datetime features extracted: {len(self.stats['extracted_datetime_features'])}",
        ])
        if self.stats['extracted_datetime_features']:
            lines.append(f"     * {', '.join(self.stats['extracted_datetime_features'])}")

        enc = self.stats["encoded_columns"]
        lines.append("   - Categorical Encoding:")
        if enc.get("one_hot"):
            lines.append(f"     * One-Hot Encoded: {', '.join(enc['one_hot'])}")
        if enc.get("label_encoded"):
            lines.append(f"     * Label Encoded: {', '.join(enc['label_encoded'])}")
        if enc.get("frequency_encoded"):
            lines.append(f"     * Frequency Encoded: {', '.join(enc['frequency_encoded'])}")
        if not enc.get("one_hot") and not enc.get("label_encoded") and not enc.get("frequency_encoded"):
            lines.append("     * None")

        lines.append(f"   - Scaled Features ({len(self.stats['scaled_columns'])}):")
        if self.stats["scaled_columns"]:
            lines.append(f"     * {', '.join(self.stats['scaled_columns'])}")
        else:
            lines.append("     * None")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_report(self, path: str | Path) -> None:
        """Save report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def save_summary_text(self, path: str | Path) -> None:
        """Save text summary to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate_text_summary())
