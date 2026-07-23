import os
import unittest

from fastapi.testclient import TestClient

from app import app


class AppApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "API Running Successfully")

    def test_predict_endpoint_reports_missing_model_gracefully(self):
        sample_path = os.path.join("tests", "testdata", "sample.jpg")
        with open(sample_path, "wb") as f:
            f.write(b"not-a-real-image")

        with open(sample_path, "rb") as sample_file:
            response = self.client.post(
                "/predict",
                files={"file": ("sample.jpg", sample_file, "image/jpeg")},
            )

        self.assertIn(response.status_code, [200, 503, 400])
        self.assertIsInstance(response.json(), dict)

    def test_history_endpoint_returns_json(self):
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), dict)
        self.assertIn("records", response.json())


if __name__ == "__main__":
    unittest.main()
