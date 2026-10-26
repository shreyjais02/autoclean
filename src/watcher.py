"""
File watcher that automatically pre-processes CSV datasets dropped into an input directory.
"""

import time
import os
from pathlib import Path
from src.config import PreprocessingConfig
from src.pipeline import PreprocessingPipeline


def start_directory_watcher(
    input_dir: str = "data/input",
    output_dir: str = "data/output",
    config: PreprocessingConfig = None,
    poll_interval: float = 2.0,
):
    """
    Watches `input_dir` for new .csv files, automatically preprocesses them,
    and outputs the cleaned file + reports into `output_dir`.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    in_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    config = config or PreprocessingConfig()

    processed_files = set()
    print("=" * 60)
    print(f" Watching for CSV files in: {in_path.resolve()}")
    print(f" Outputs will be saved to:  {out_path.resolve()}")
    print(" Press Ctrl+C to stop watching.")
    print("=" * 60)

    try:
        while True:
            csv_files = list(in_path.glob("*.csv"))
            for csv_file in csv_files:
                # Check if file is ready and not already processed
                file_key = (csv_file.name, csv_file.stat().st_mtime)
                if file_key not in processed_files:
                    # Brief pause to ensure file write is finished
                    time.sleep(0.5)
                    print(f"\n[NEW FILE DETECTED] Processing: {csv_file.name} ...")
                    try:
                        pipeline = PreprocessingPipeline(config=config)
                        out_file = out_path / f"{csv_file.stem}_preprocessed.csv"
                        pipeline.process_file(csv_file, output_csv_path=out_file, save_report=True)
                        print(f"[DONE] Successfully preprocessed -> {out_file.name}")
                        print(pipeline.reporter.generate_text_summary())
                        processed_files.add(file_key)
                    except Exception as e:
                        print(f"[ERROR] Failed to process {csv_file.name}: {e}")
                        processed_files.add(file_key)

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nWatcher stopped by user.")
