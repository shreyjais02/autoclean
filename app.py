"""
Flask Web Application for Automated CSV Data Preprocessing.
"""

import sys
import os
import uuid
import json
import time
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PreprocessingConfig
from src.pipeline import PreprocessingPipeline
from examples.generate_sample_data import generate_dirty_dataset

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max upload

UPLOAD_DIR = Path("data/web_uploads")
OUTPUT_DIR = Path("data/web_outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    """Render the web UI."""
    return render_template("index.html")


@app.route("/api/load-sample", methods=["POST"])
def load_sample():
    """Generates a sample dirty dataset and returns preview information."""
    try:
        task_id = str(uuid.uuid4())[:8]
        sample_path = UPLOAD_DIR / f"sample_{task_id}.csv"
        generate_dirty_dataset(n_rows=300, output_path=sample_path)

        df = pd.read_csv(sample_path)
        preview_data = df.head(10).astype(object).fillna("").to_dict(orient="records")

        return jsonify({
            "status": "success",
            "task_id": task_id,
            "filename": "sample_dirty_customers.csv",
            "file_path": str(sample_path),
            "rows": len(df),
            "columns": list(df.columns),
            "missing_total": int(df.isna().sum().sum()),
            "preview": preview_data,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handles CSV upload and returns metadata and initial preview."""
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file part in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"status": "error", "message": "Only CSV files are currently supported"}), 400

        task_id = str(uuid.uuid4())[:8]
        saved_name = f"{task_id}_{file.filename}"
        saved_path = UPLOAD_DIR / saved_name
        file.save(saved_path)

        # Inspect using smart CSV loader
        df = PreprocessingPipeline.read_csv_smart(saved_path)
        preview_data = df.head(10).astype(object).fillna("").to_dict(orient="records")

        return jsonify({
            "status": "success",
            "task_id": task_id,
            "filename": file.filename,
            "file_path": str(saved_path),
            "rows": len(df),
            "columns": list(df.columns),
            "missing_total": int(df.isna().sum().sum()),
            "preview": preview_data,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error reading file: {str(e)}"}), 500


@app.route("/api/process", methods=["POST"])
def process_dataset():
    """Preprocesses dataset with custom or auto configurations."""
    try:
        data = request.json or {}
        file_path_str = data.get("file_path")
        if not file_path_str:
            return jsonify({"status": "error", "message": "No file path provided"}), 400

        file_path = Path(file_path_str)
        if not file_path.exists():
            return jsonify({"status": "error", "message": "Source dataset file not found"}), 404

        # Read config options from request
        target_col = data.get("target_column") or None
        scaling = data.get("scaling_strategy", "standard")
        encoding = data.get("encoding_strategy", "auto")
        outlier_method = data.get("outlier_method", "iqr")
        outlier_action = data.get("outlier_action", "clip")
        drop_duplicates = bool(data.get("drop_duplicates", True))
        drop_constant = bool(data.get("drop_constant_columns", True))
        extract_datetime = bool(data.get("extract_datetime_features", True))

        config = PreprocessingConfig(
            target_column=target_col if target_col != "" else None,
            scaling_strategy=scaling,
            encoding_strategy=encoding,
            outlier_method=outlier_method,
            outlier_action=outlier_action,
            drop_duplicates=drop_duplicates,
            drop_constant_columns=drop_constant,
            extract_datetime_features=extract_datetime,
        )

        task_id = data.get("task_id") or str(uuid.uuid4())[:8]
        out_csv_path = OUTPUT_DIR / f"{task_id}_preprocessed.csv"
        out_report_path = OUTPUT_DIR / f"{task_id}_preprocessed_report.json"
        out_model_path = OUTPUT_DIR / f"{task_id}_preprocessed_pipeline.joblib"

        pipeline = PreprocessingPipeline(config=config)
        pipeline.process_file(
            input_csv_path=file_path,
            output_csv_path=out_csv_path,
            save_report=True,
            save_pipeline=True,
        )

        # Load processed preview
        df_processed = pd.read_csv(out_csv_path)
        processed_preview = df_processed.head(15).astype(object).fillna("").to_dict(orient="records")

        # Load report
        with open(out_report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        return jsonify({
            "status": "success",
            "task_id": task_id,
            "output_rows": len(df_processed),
            "output_columns_count": len(df_processed.columns),
            "output_columns": list(df_processed.columns),
            "preview": processed_preview,
            "report": report_data,
            "summary_text": pipeline.reporter.generate_text_summary(),
            "download_csv_url": f"/api/download/csv/{task_id}",
            "download_report_url": f"/api/download/report/{task_id}",
            "download_pipeline_url": f"/api/download/pipeline/{task_id}",
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Processing error: {str(e)}"}), 500


@app.route("/api/download/<file_type>/<task_id>", methods=["GET"])
def download_file(file_type: str, task_id: str):
    """Serve processed downloads."""
    try:
        if file_type == "csv":
            target_path = OUTPUT_DIR / f"{task_id}_preprocessed.csv"
            return send_file(
                target_path,
                as_attachment=True,
                download_name=f"dataset_preprocessed_{task_id}.csv",
                mimetype="text/csv",
            )
        elif file_type == "report":
            target_path = OUTPUT_DIR / f"{task_id}_preprocessed_report.json"
            return send_file(
                target_path,
                as_attachment=True,
                download_name=f"audit_report_{task_id}.json",
                mimetype="application/json",
            )
        elif file_type == "pipeline":
            target_path = OUTPUT_DIR / f"{task_id}_preprocessed_pipeline.joblib"
            return send_file(
                target_path,
                as_attachment=True,
                download_name=f"preprocessor_model_{task_id}.joblib",
                mimetype="application/octet-stream",
            )
        else:
            return jsonify({"status": "error", "message": "Invalid file type"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" 🌟 Web Preprocessing UI running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=True)
