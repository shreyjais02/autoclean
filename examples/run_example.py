"""
End-to-End Demonstration of the Automated Preprocessing Pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import PreprocessingConfig
from src.pipeline import PreprocessingPipeline
from examples.generate_sample_data import generate_dirty_dataset


def run_demo():
    print("Step 1: Generating a dirty sample CSV dataset...")
    raw_csv = generate_dirty_dataset(n_rows=400, output_path="data/input/customers_dirty.csv")

    print("\nStep 2: Inspecting first 5 rows of raw data:")
    df_raw = pd.read_csv(raw_csv)
    print(df_raw.head())

    print("\nStep 3: Initializing Preprocessing Pipeline...")
    config = PreprocessingConfig(
        target_column="churn",
        scaling_strategy="standard",
        encoding_strategy="auto",
        outlier_method="iqr",
        outlier_action="clip",
    )

    pipeline = PreprocessingPipeline(config=config)

    print("\nStep 4: Processing dataset...")
    output_path = pipeline.process_file(
        input_csv_path=raw_csv,
        output_csv_path="data/output/customers_cleaned.csv",
        save_report=True,
        save_pipeline=True,
    )

    print(pipeline.reporter.generate_text_summary())

    print("\nStep 5: Inspecting first 5 rows of PREPROCESSED data:")
    df_cleaned = pd.read_csv(output_path)
    print(df_cleaned.head())
    print("\nCleaned columns:", list(df_cleaned.columns))
    print(f"\nAll outputs delivered to: {output_path.parent.resolve()}")


if __name__ == "__main__":
    run_demo()
