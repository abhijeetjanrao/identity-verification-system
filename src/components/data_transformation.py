import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis


class DataTransformation:

    def __init__(self, dataset_path, metadata_path, artifact_dir):
        self.dataset_path = dataset_path
        self.metadata_path = metadata_path
        self.artifact_dir = artifact_dir

        os.makedirs(self.artifact_dir, exist_ok=True)

    def initiate_transformation(self):

        print("Loading metadata...")
        metadata = pd.read_csv(self.metadata_path)

        print("Initializing InsightFace...")
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))  # Balanced detection size

        person_embeddings = {}  # Dictionary: {person_id: [embeddings]}

        for folder in os.listdir(self.dataset_path):

            folder_path = os.path.join(self.dataset_path, folder)

            if not os.path.isdir(folder_path):
                continue

            row = metadata[metadata["folder_name"] == folder]
            if row.empty:
                continue

            person_id = int(row["person_id"].values[0])

            if person_id not in person_embeddings:
                person_embeddings[person_id] = []

            for image_name in os.listdir(folder_path):

                image_path = os.path.join(folder_path, image_name)

                img = cv2.imread(image_path)
                if img is None:
                    continue

                # 🔥 Upscale very small images
                h, w = img.shape[:2]
                if min(h, w) < 250:
                    img = cv2.resize(img, None, fx=2, fy=2)

                faces = app.get(img)

                if len(faces) == 0:
                    continue

                embedding = faces[0].embedding

                # ✅ Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)

                person_embeddings[person_id].append(embedding)

        print("Calculating mean embeddings per person...")

        final_embeddings = []
        final_labels = []

        for person_id, embeddings in person_embeddings.items():

            if len(embeddings) == 0:
                continue

            # 🔥 Compute mean embedding
            mean_embedding = np.mean(embeddings, axis=0)

            # Normalize mean embedding again
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

            final_embeddings.append(mean_embedding)
            final_labels.append(person_id)

        final_embeddings = np.array(final_embeddings).astype("float32")
        final_labels = np.array(final_labels)

        np.save(os.path.join(self.artifact_dir, "embeddings.npy"), final_embeddings)
        np.save(os.path.join(self.artifact_dir, "labels.npy"), final_labels)

        print("Transformation completed.")
        print("Total identities stored:", len(final_embeddings))

        return final_embeddings, final_labels