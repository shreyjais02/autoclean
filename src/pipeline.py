"""
Unified automated PreprocessingPipeline orchestrator.
"""

from typing import Optional, Union, Dict, Any
import time
from pathlib import Path
import pandas as pd
import joblib

from src.config import PreprocessingConfig
from src.detector import ColumnDetector
from src.cleaner import DataCleaner
from src.transformer import FeatureTransformer
from src.reporter import PreprocessingReporter


class PreprocessingPipeline:
    """
    End-to-end Automated Preprocessing Pipeline.
    
    Can be used directly on DataFrames or with CSV files.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.detector = ColumnDetector(self.config)
        self.cleaner = DataCleaner(self.config)
        self.transformer = FeatureTransformer(self.config)
        self.reporter = PreprocessingReporter()
        self.detected_schema_: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """Fit detector, cleaner, and transformer on input data."""
        self.reporter.record_input(df)
        self.detected_schema_ = self.detector.detect(df)
        self.reporter.record_detections(self.detected_schema_)

        # Fit cleaner
        self.cleaner.fit(df, self.detected_schema_)
        df_cleaned = self.cleaner.transform(df, self.detected_schema_)
        self.reporter.record_cleaning(self.cleaner)

        # Fit transformer
        self.transformer.fit(df_cleaned, self.detected_schema_)
        self.reporter.record_transformations(self.transformer)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline rules."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        df_cleaned = self.cleaner.transform(df, self.detected_schema_)
        df_transformed = self.transformer.transform(df_cleaned)
        return df_transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit pipeline and return transformed DataFrame."""
        start_time = time.time()
        self.fit(df)
        df_out = self.transform(df)
        elapsed = time.time() - start_time
        self.reporter.record_output(df_out, elapsed)
        return df_out

    def process_file(
        self,
        input_csv_path: Union[str, Path],
        output_csv_path: Optional[Union[str, Path]] = None,
        save_report: bool = True,
        save_pipeline: bool = False,
    ) -> Path:
        """
        Reads CSV file from input path, runs the pipeline, and saves results.

        Parameters
        ----------
        input_csv_path : str or Path
            Path to raw input CSV file.
        output_csv_path : str or Path, optional
            Path where preprocessed CSV will be saved. Default is 'data/output/<name>_preprocessed.csv'.
        save_report : bool
            Whether to save JSON audit report and text summary.
        save_pipeline : bool
            Whether to serialize the fitted pipeline (.pkl).

        Returns
        -------
        Path
            Path to the saved preprocessed CSV file.
        """
        input_path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Smart CSV ingestion
        df_raw = self.read_csv_smart(input_path)

        # Execute preprocessing
        df_processed = self.fit_transform(df_raw)

        # Determine output paths
        if output_csv_path is None:
            out_dir = Path("data/output")
            out_dir.mkdir(parents=True, exist_ok=True)
            output_csv_path = out_dir / f"{input_path.stem}_preprocessed.csv"
        else:
            output_csv_path = Path(output_csv_path)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Save preprocessed dataset
        df_processed.to_csv(output_csv_path, index=False)

        # Save reports
        if save_report:
            report_json = output_csv_path.parent / f"{output_csv_path.stem}_report.json"
            report_txt = output_csv_path.parent / f"{output_csv_path.stem}_summary.txt"
            self.reporter.save_report(report_json)
            self.reporter.save_summary_text(report_txt)

        # Save fitted pipeline
        if save_pipeline:
            pipeline_path = output_csv_path.parent / f"{output_csv_path.stem}_pipeline.joblib"
            self.save(pipeline_path)

        return output_csv_path

    @staticmethod
    def read_csv_smart(path: Union[str, Path]) -> pd.DataFrame:
        """Read CSV handling various delimiters and encodings."""
        encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
        delimiters = [",", ";", "\t", "|"]

        last_err = None
        for enc in encodings:
            for sep in delimiters:
                try:
                    df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                    if df.shape[1] > 1:
                        return df
                except Exception as e:
                    last_err = e
                    continue

        # Fallback standard read
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            raise RuntimeError(f"Failed to read CSV at {path}: {last_err}")

    def save(self, path: Union[str, Path]) -> None:
        """Serialize fitted pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PreprocessingPipeline":
        """Load fitted pipeline from disk."""
        return joblib.load(path)
