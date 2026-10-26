#!/usr/bin/env python3
"""
CLI entry point for the Automated Data Preprocessing Pipeline.
Usage:
    python preprocess.py dataset.csv
    python preprocess.py dataset.csv --output cleaned.csv --target price --scaling minmax
    python preprocess.py --watch
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PreprocessingConfig
from src.pipeline import PreprocessingPipeline
from src.watcher import start_directory_watcher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Python-based Data Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=str,
        help="Path to the input CSV file to preprocess.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Custom output CSV path. Defaults to data/output/<name>_preprocessed.csv",
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        default=None,
        help="Target column name (will not be scaled/encoded as feature).",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--scaling",
        type=str,
        choices=["standard", "minmax", "robust", "none"],
        default=None,
        help="Numerical scaling technique.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["auto", "onehot", "label", "frequency", "none"],
        default=None,
        help="Categorical encoding technique.",
    )
    parser.add_argument(
        "--outliers",
        type=str,
        choices=["iqr", "zscore", "none"],
        default=None,
        help="Outlier detection method.",
    )
    parser.add_argument(
        "--save-pipeline",
        action="store_true",
        help="Save fitted pipeline as a serialized .joblib file for future inference.",
    )
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Watch data/input directory and automatically process any newly added CSV files.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/input",
        help="Directory to watch when --watch mode is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Directory to save outputs when --watch mode is enabled.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # If watcher mode
    if args.watch:
        config = PreprocessingConfig.from_yaml(args.config) if args.config else PreprocessingConfig()
        if args.target:
            config.target_column = args.target
        start_directory_watcher(args.input_dir, args.output_dir, config)
        return

    # Check input CSV
    if not args.input_csv:
        print("\n[Error] No input CSV specified.")
        print("Usage: python preprocess.py <path_to_csv> [options]")
        print("Run 'python preprocess.py --help' for details, or 'python preprocess.py --watch' to watch folder.")
        sys.exit(1)

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"\n[Error] Input file not found: {input_path}")
        sys.exit(1)

    # Build config
    if args.config:
        config = PreprocessingConfig.from_yaml(args.config)
    else:
        config = PreprocessingConfig()

    if args.target is not None:
        config.target_column = args.target
    if args.scaling is not None:
        config.scaling_strategy = args.scaling
    if args.encoding is not None:
        config.encoding_strategy = args.encoding
    if args.outliers is not None:
        config.outlier_method = args.outliers

    print("\n" + "=" * 60)
    print("      STARTING AUTOMATED PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f" Input File: {input_path.resolve()}")

    pipeline = PreprocessingPipeline(config=config)
    output_path = pipeline.process_file(
        input_csv_path=input_path,
        output_csv_path=args.output,
        save_report=True,
        save_pipeline=args.save_pipeline,
    )

    # Print summary
    print(pipeline.reporter.generate_text_summary())
    print(f"\n[SUCCESS] Preprocessed dataset saved to: {output_path.resolve()}")
    print(f"[SUCCESS] Audit report saved to:        {output_path.parent / (output_path.stem + '_report.json')}")
    print(f"[SUCCESS] Summary text saved to:        {output_path.parent / (output_path.stem + '_summary.txt')}\n")


if __name__ == "__main__":
    main()
