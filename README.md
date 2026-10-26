# 🚀 AutoClean: Intelligent Tabular Data Preprocessing Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" alt="Python Versions">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Framework-Flask%20%7C%20Scikit--Learn%20%7C%20Pandas-orange.svg" alt="Frameworks">
  <img src="https://img.shields.io/badge/Tests-Passing%20(9%2F9)-brightgreen.svg" alt="Tests">
</p>

An end-to-end, automated Python data preprocessing engine with a **modern web UI (HTML5 & CSS3)**, **CLI**, and **Python API**. Drop in any raw CSV dataset, and AutoClean automatically detects types, imputes missing values, cleans formatted numbers, clips outliers, decomposes datetimes, encodes categoricals, scales features, and delivers clean ML-ready data with audit reports.

---

## 🌟 Key Features

- 🌐 **Modern Web UI**:
  - Drag-and-drop CSV upload with auto-detection of delimiters (`,`, `;`, `\t`, `|`) and encodings (`utf-8`, `latin1`).
  - 1-Click instant test with built-in dirty sample dataset generation.
  - Interactive tuning: Target column protection, Scaling (Standard, MinMax, Robust), Encoding (Auto, One-Hot, Label, Frequency), Outlier handling (IQR, Z-Score).
  - Tabbed live data preview and downloadable artifacts: **Cleaned CSV**, **Audit Report (JSON)**, and **Fitted Model (`.joblib`)**.
- 🔍 **Intelligent Detection**:
  - Distinguishes continuous numbers, currency/percentage formatted strings, categoricals, datetimes, IDs, constants, and high-missing columns.
- 🧹 **Robust Data Cleaning**:
  - Drops exact duplicates and uninformative columns.
  - Imputes numerical missing values (median/mean/constant) and categorical missing values (mode/constant).
  - Outlier detection and clipping via IQR (Interquartile Range) or Z-score boundaries.
- ⚙️ **Feature Engineering**:
  - Automatic datetime extraction (`year`, `month`, `day`, `day_of_week`, `is_weekend`, `hour`).
  - Automatic cardinality-aware categorical encoding (One-Hot for low cardinality, Label/Frequency for high cardinality).
  - Numerical scaling with target column isolation.
- 💾 **Pipeline Serialization**:
  - Saves fitted pipelines with `joblib` for re-use on unseen test/production data without data leakage.
- 🐳 **Docker & CI/CD Ready**:
  - Includes `Dockerfile`, `docker-compose.yml`, and GitHub Actions CI workflow across Python 3.10–3.13.

---

## 📂 Repository Structure

```text
autoclean-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI matrix test runner
├── data/
│   ├── input/                  # Folder for input CSVs
│   └── output/                 # Folder for preprocessed CSVs & reports
├── examples/
│   ├── generate_sample_data.py # Realistic dirty dataset generator
│   └── run_example.py          # End-to-end Python script demonstration
├── src/                        # Core Preprocessing Package
│   ├── __init__.py
│   ├── cleaner.py              # Deduplication, imputation, outlier treatment
│   ├── config.py               # Preprocessing configuration dataclass & YAML
│   ├── detector.py             # Automatic column quality & semantic type detector
│   ├── pipeline.py             # PreprocessingPipeline orchestrator (fit/transform)
│   ├── reporter.py             # Human-readable & JSON audit generator
│   ├── transformer.py          # Encoding, scaling & datetime decomposition
│   └── watcher.py              # Automatic folder watcher daemon
├── static/
│   ├── css/style.css           # Modern dark-theme stylesheet
│   └── js/app.js               # Client-side UI logic, drag & drop, data tables
├── templates/
│   └── index.html              # HTML5 Web UI template
├── tests/
│   ├── test_pipeline.py        # Core pipeline unit tests
│   └── test_web.py             # Flask Web UI integration tests
├── app.py                      # Flask Web server & REST API
├── preprocess.py               # CLI runner & directory watcher
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Modern Python packaging configuration
├── Dockerfile                  # Containerization image
├── docker-compose.yml          # Container composition
├── LICENSE                     # MIT License
└── CONTRIBUTING.md             # Contribution guidelines
```

---

## 🚀 Quickstart

### 1. Installation

Clone your repository and install dependencies:

```bash
git clone https://github.com/<your-username>/autoclean-pipeline.git
cd autoclean-pipeline
pip install -r requirements.txt
```

---

### 2. Launch the Web UI

```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser. Drag and drop any CSV or click **"⚡ Load Sample Messy Dataset"** to run!

---

### 3. Docker (Optional)

Run the full web application in a container with a single command:

```bash
docker-compose up --build
```
Access at **`http://localhost:5000`**.

---

### 4. CLI Usage

Preprocess any CSV directly from your terminal:

```bash
# Basic one-liner
python preprocess.py data/input/my_dataset.csv

# Custom options with target protection and MinMax scaling
python preprocess.py data/input/my_dataset.csv --target churn --scaling minmax --save-pipeline

# Automated folder watcher (processes any file dropped in data/input)
python preprocess.py --watch
```

---

### 5. Python API Usage

```python
import pandas as pd
from src.config import PreprocessingConfig
from src.pipeline import PreprocessingPipeline

# 1. Initialize configuration
config = PreprocessingConfig(
    target_column="churn",
    scaling_strategy="standard",  # 'standard', 'minmax', 'robust', 'none'
    encoding_strategy="auto",     # 'auto', 'onehot', 'label', 'frequency'
    outlier_method="iqr",         # 'iqr', 'zscore', 'none'
    outlier_action="clip",
)

# 2. Run pipeline
pipeline = PreprocessingPipeline(config=config)
output_file = pipeline.process_file("data/input/customers.csv")

# 3. View text summary audit
print(pipeline.reporter.generate_text_summary())
```

---

## 🧪 Testing

Run all unit and integration tests:

```bash
python -m unittest discover -s tests -v
```

---

## 📤 How to Push to Your GitHub

Follow these steps in your terminal:

```bash
# 1. Initialize git
git init

# 2. Stage all files
git add .

# 3. Commit
git commit -m "feat: initial release of AutoClean automated preprocessing pipeline"

# 4. Set default branch to main
git branch -M main

# 5. Link your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/<your-username>/autoclean-pipeline.git

# 6. Push to GitHub
git push -u origin main
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
