"""
Unit & Integration tests for the Flask Web UI Application.
"""

import sys
import unittest
import io
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app


class TestWebApp(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_index_page(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"AutoClean", res.data)
        self.assertIn(b"drop-zone", res.data)

    def test_load_sample(self):
        res = self.client.post("/api/load-sample")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("task_id", data)
        self.assertGreater(data["rows"], 0)
        self.assertIn("columns", data)
        self.assertIn("preview", data)

    def test_upload_and_process_flow(self):
        # 1. Test CSV upload
        csv_content = (
            "id,name,age,salary,country,churn\n"
            "1,Alice,25,$50000,USA,0\n"
            "2,Bob,30,$60000,UK,1\n"
            "3,Charlie,,$75000,USA,0\n"
            "4,David,200,$80000,Canada,1\n"
            "4,David,200,$80000,Canada,1\n"  # Duplicate
        )

        data = {
            "file": (io.BytesIO(csv_content.encode("utf-8")), "test_customers.csv")
        }

        upload_res = self.client.post(
            "/api/upload",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(upload_res.status_code, 200)
        upload_data = upload_res.get_json()
        self.assertEqual(upload_data["status"], "success")
        task_id = upload_data["task_id"]
        file_path = upload_data["file_path"]

        # 2. Test Processing
        process_payload = {
            "task_id": task_id,
            "file_path": file_path,
            "target_column": "churn",
            "scaling_strategy": "standard",
            "encoding_strategy": "auto",
            "outlier_method": "iqr",
            "outlier_action": "clip",
            "drop_duplicates": True,
            "drop_constant_columns": True,
            "extract_datetime_features": True,
        }

        process_res = self.client.post(
            "/api/process",
            data=json.dumps(process_payload),
            content_type="application/json"
        )
        self.assertEqual(process_res.status_code, 200)
        process_data = process_res.get_json()
        self.assertEqual(process_data["status"], "success")
        self.assertEqual(process_data["output_rows"], 4)  # 5 rows - 1 duplicate = 4 rows
        self.assertIn("download_csv_url", process_data)

        # 3. Test Downloads
        csv_res = self.client.get(process_data["download_csv_url"])
        self.assertEqual(csv_res.status_code, 200)
        self.assertTrue(len(csv_res.data) > 0)

        report_res = self.client.get(process_data["download_report_url"])
        self.assertEqual(report_res.status_code, 200)

        model_res = self.client.get(process_data["download_pipeline_url"])
        self.assertEqual(model_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
