import os

import numpy as np
import pandas as pd


class PredictionPipeline:

    def __init__(self):
        self.artifact_dir = "saved_models"
        self.index = None
        self.labels = None
        self.metadata = None
        self.app = None
        self.ready = False
        self.error = None
        self._initialize()

    def _initialize(self):
        try:
            import cv2
            import faiss
            from insightface.app import FaceAnalysis
        except Exception as exc:
            self.error = f"Missing runtime dependency: {exc}"
            return

        index_path = os.path.join(self.artifact_dir, "faiss_index.bin")
        labels_path = os.path.join(self.artifact_dir, "labels.npy")
        metadata_path = "metadata.csv"

        if not os.path.exists(index_path):
            self.error = f"Missing model artifact: {index_path}"
            return
        if not os.path.exists(labels_path):
            self.error = f"Missing model artifact: {labels_path}"
            return
        if not os.path.exists(metadata_path):
            self.error = f"Missing metadata file: {metadata_path}"
            return

        try:
            self.index = faiss.read_index(index_path)
            self.labels = np.load(labels_path)
            self.metadata = pd.read_csv(metadata_path)
        except Exception as exc:
            self.error = f"Failed to load model assets: {exc}"
            return

        try:
            self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            try:
                self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                self.app.prepare(ctx_id=0, det_size=(640, 640))
            except Exception as exc:
                self.error = f"Could not initialize face model: {exc}"
                return

        self.ready = True

    def predict(self, image_path):
        if not self.ready:
            return {"error": self.error or "Model not ready", "status": "unavailable"}

        import cv2

        img = cv2.imread(image_path)

        if img is None:
            return {"message": "Invalid image path", "status": "invalid_image"}

        h, w = img.shape[:2]
        if min(h, w) < 250:
            img = cv2.resize(img, None, fx=2, fy=2)

        faces = self.app.get(img)

        if len(faces) == 0:
            return {"message": "No face detected", "status": "no_face"}

        embedding = faces[0].embedding
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.astype("float32")

        D, I = self.index.search(embedding.reshape(1, -1), 1)

        similarity = float(D[0][0])
        matched_label = self.labels[I[0][0]]

        if similarity < 0.60:
            return {
                "name": "Unknown",
                "profession": None,
                "description": None,
                "confidence": similarity,
                "status": "unknown",
            }

        person_row = self.metadata[self.metadata["person_id"] == matched_label]
        if person_row.empty:
            return {
                "name": "Unknown",
                "profession": None,
                "description": None,
                "confidence": similarity,
                "status": "unknown",
            }

        return {
            "name": person_row["name"].values[0],
            "profession": person_row["profession"].values[0],
            "description": person_row["description"].values[0],
            "confidence": similarity,
            "status": "matched",
        }