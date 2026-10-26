"""
Unit and integration test suite for the Automated Preprocessing Pipeline.
"""

import sys
import unittest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PreprocessingConfig
from src.detector import ColumnDetector
from src.cleaner import DataCleaner
from src.transformer import FeatureTransformer
from src.pipeline import PreprocessingPipeline


class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        # Create a sample dirty test DataFrame
        self.df = pd.DataFrame({
            "id": ["A1", "A2", "A3", "A4", "A5", "A5"],
            "age": [25, np.nan, 30, 150, 40, 40],  # Outlier (150) & missing
            "salary": ["$50,000", "$60,000", np.nan, "$75,000", "$80,000", "$80,000"],
            "city": ["New York", "London", "Paris", "New York", "London", "London"],
            "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-05-01"],
            "const_col": ["Same", "Same", "Same", "Same", "Same", "Same"],
            "empty_col": [np.nan, np.nan, np.nan, np.nan, "val", np.nan],  # >80% missing
            "target": [0, 1, 0, 1, 0, 0],
        })

    def test_detector(self):
        config = PreprocessingConfig(target_column="target")
        detector = ColumnDetector(config)
        detected = detector.detect(self.df)

        self.assertIn("const_col", detected["constant"])
        self.assertIn("empty_col", detected["high_missing"])
        self.assertIn("id", detected["id_columns"])
        self.assertIn("age", detected["numeric"])
        self.assertIn("salary", detected["numeric"])
        self.assertIn("city", detected["categorical"])
        self.assertIn("date", detected["datetime"])

    def test_cleaner_dedup_and_imputation(self):
        config = PreprocessingConfig(target_column="target", outlier_action="clip")
        detector = ColumnDetector(config)
        detected = detector.detect(self.df)

        cleaner = DataCleaner(config)
        df_cleaned = cleaner.fit_transform(self.df, detected)

        # Duplicate row should be removed (from 6 to 5 rows)
        self.assertEqual(len(df_cleaned), 5)
        # Dropped columns should not exist
        self.assertNotIn("const_col", df_cleaned.columns)
        self.assertNotIn("empty_col", df_cleaned.columns)
        self.assertNotIn("id", df_cleaned.columns)
        # Missing values in age and salary should be resolved
        self.assertEqual(df_cleaned["age"].isna().sum(), 0)
        self.assertEqual(df_cleaned["salary"].isna().sum(), 0)

    def test_end_to_end_pipeline(self):
        config = PreprocessingConfig(
            target_column="target",
            scaling_strategy="standard",
            encoding_strategy="onehot",
            outlier_action="clip",
        )
        pipeline = PreprocessingPipeline(config)
        df_out = pipeline.fit_transform(self.df)

        # Output checks
        self.assertFalse(df_out.isna().any().any(), "No missing values should remain.")
        self.assertIn("target", df_out.columns, "Target column must be retained.")
        self.assertIn("date_year", df_out.columns, "Datetime features should be extracted.")
        self.assertTrue(any(c.startswith("city_") for c in df_out.columns), "One-hot dummy columns should exist.")

    def test_file_processing_and_serialization(self):
        config = PreprocessingConfig(target_column="target")
        pipeline = PreprocessingPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "test_input.csv"
            output_csv = Path(tmpdir) / "test_output.csv"
            model_path = Path(tmpdir) / "model.joblib"

            self.df.to_csv(input_csv, index=False)

            pipeline.process_file(input_csv, output_csv_path=output_csv, save_pipeline=True)
            self.assertTrue(output_csv.exists())

            # Test pipeline serialization & loading
            pipeline.save(model_path)
            self.assertTrue(model_path.exists())

            loaded_pipeline = PreprocessingPipeline.load(model_path)
            self.assertTrue(loaded_pipeline.is_fitted)

            # Test inference on single new row
            new_df = pd.DataFrame([{
                "id": "A99",
                "age": 32,
                "salary": "$55,000",
                "city": "London",
                "date": "2023-06-01",
                "const_col": "Same",
                "empty_col": np.nan,
                "target": 1,
            }])
            new_transformed = loaded_pipeline.transform(new_df)
            self.assertEqual(len(new_transformed), 1)
            self.assertEqual(new_transformed.isna().sum().sum(), 0)

    def test_encoding_and_scaling_variants(self):
        for enc in ["label", "frequency"]:
            for scale in ["minmax", "robust", "none"]:
                config = PreprocessingConfig(
                    target_column="target",
                    encoding_strategy=enc,
                    scaling_strategy=scale,
                )
                pipe = PreprocessingPipeline(config)
                df_out = pipe.fit_transform(self.df)
                self.assertFalse(df_out.isna().any().any())

    def test_smart_csv_delimiters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for sep in [";", "\t", "|"]:
                p = Path(tmpdir) / f"test_{ord(sep)}.csv"
                self.df.to_csv(p, sep=sep, index=False)
                df_read = PreprocessingPipeline.read_csv_smart(p)
                self.assertEqual(df_read.shape[1], self.df.shape[1])


if __name__ == "__main__":
    unittest.main()
