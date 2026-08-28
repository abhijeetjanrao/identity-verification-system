```python
import os
from pathlib import Path

import numpy as np
import pandas as pd


class PredictionPipeline:

    def __init__(self):
        # -------------------------------------------------
        # Project root
        #
        # identity-verification-system/
        # ├── app.py
        # ├── src/
        # │   └── pipeline/
        # │       └── prediction_pipeline.py
        # ├── saved_models/
        # │   ├── faiss_index.bin
        # │   └── labels.npy
        # └── metadata.csv
        # -------------------------------------------------
        self.base_dir = Path(__file__).resolve().parents[2]

        self.artifact_dir = self.base_dir / "saved_models"

        self.index = None
        self.labels = None
        self.metadata = None
        self.app = None

        self.ready = False
        self.error = None

        self._initialize()

    # =====================================================
    # INITIALIZATION
    # =====================================================
    def _initialize(self):

        # -------------------------------------------------
        # Check runtime dependencies
        # -------------------------------------------------
        try:
            import cv2
            import faiss
            from insightface.app import FaceAnalysis
        except Exception as exc:
            self.error = f"Missing runtime dependency: {exc}"
            return

        # -------------------------------------------------
        # Model artifact paths
        # -------------------------------------------------
        index_path = self.artifact_dir / "faiss_index.bin"
        labels_path = self.artifact_dir / "labels.npy"
        metadata_path = self.base_dir / "metadata.csv"

        # -------------------------------------------------
        # Check required files
        # -------------------------------------------------
        if not index_path.exists():
            self.error = f"Missing model artifact: {index_path}"
            return

        if not labels_path.exists():
            self.error = f"Missing model artifact: {labels_path}"
            return

        if not metadata_path.exists():
            self.error = f"Missing metadata file: {metadata_path}"
            return

        # -------------------------------------------------
        # Load FAISS index
        # -------------------------------------------------
        try:
            self.index = faiss.read_index(str(index_path))
            self.labels = np.load(str(labels_path))
            self.metadata = pd.read_csv(str(metadata_path))

        except Exception as exc:
            self.error = f"Failed to load model assets: {exc}"
            return

        # -------------------------------------------------
        # Initialize InsightFace
        #
        # IMPORTANT:
        # Render is using CPU.
        #
        # Do NOT use:
        # CUDAExecutionProvider
        #
        # Do NOT use:
        # ctx_id=0
        #
        # Use:
        # CPUExecutionProvider
        # ctx_id=-1
        # -------------------------------------------------
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )

            self.app.prepare(
                ctx_id=-1,
                det_size=(640, 640)
            )

        except Exception as exc:
            self.error = f"Could not initialize face model: {exc}"
            return

        # -------------------------------------------------
        # Everything initialized successfully
        # -------------------------------------------------
        self.ready = True
        self.error = None

    # =====================================================
    # PREDICTION
    # =====================================================
    def predict(self, image_path):

        if not self.ready:
            return {
                "error": self.error or "Model not ready",
                "status": "unavailable"
            }

        import cv2

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------
        img = cv2.imread(image_path)

        if img is None:
            return {
                "message": "Invalid image path",
                "status": "invalid_image"
            }

        # -------------------------------------------------
        # Resize small images
        # -------------------------------------------------
        h, w = img.shape[:2]

        if min(h, w) < 250:
            img = cv2.resize(
                img,
                None,
                fx=2,
                fy=2
            )

        # -------------------------------------------------
        # Detect faces
        # -------------------------------------------------
        faces = self.app.get(img)

        if len(faces) == 0:
            return {
                "message": "No face detected",
                "status": "no_face"
            }

        # -------------------------------------------------
        # Get face embedding
        # -------------------------------------------------
        embedding = faces[0].embedding

        embedding = embedding / np.linalg.norm(embedding)

        embedding = embedding.astype("float32")

        # -------------------------------------------------
        # Search FAISS index
        # -------------------------------------------------
        D, I = self.index.search(
            embedding.reshape(1, -1),
            1
        )

        similarity = float(D[0][0])

        matched_label = self.labels[I[0][0]]

        # -------------------------------------------------
        # Unknown person
        # -------------------------------------------------
        if similarity < 0.60:
            return {
                "name": "Unknown",
                "profession": None,
                "description": None,
                "confidence": similarity,
                "status": "unknown"
            }

        # -------------------------------------------------
        # Find person metadata
        # -------------------------------------------------
        person_row = self.metadata[
            self.metadata["person_id"] == matched_label
        ]

        if person_row.empty:
            return {
                "name": "Unknown",
                "profession": None,
                "description": None,
                "confidence": similarity,
                "status": "unknown"
            }

        # -------------------------------------------------
        # Return matched identity
        # -------------------------------------------------
        return {
            "name": person_row["name"].values[0],
            "profession": person_row["profession"].values[0],
            "description": person_row["description"].values[0],
            "confidence": similarity,
            "status": "matched"
        }
```
