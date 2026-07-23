import os
import cv2
import faiss
import pickle
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

# -------- CONFIG --------
DATASET_PATH = "bollywood_celeb_faces_0"
METADATA_PATH = "metadata.csv"
INDEX_SAVE_PATH = "saved_models/faiss_index.bin"
MAPPING_SAVE_PATH = "saved_models/id_mapping.pkl"

os.makedirs("saved_models", exist_ok=True)

# -------- LOAD METADATA --------
metadata = pd.read_csv(METADATA_PATH)

# -------- LOAD MODEL --------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(224, 224))

embeddings = []
id_mapping = []

# -------- LOOP THROUGH DATASET --------
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    # Get metadata row
    person_row = metadata[metadata["folder_name"] == folder]
    if person_row.empty:
        continue

    person_id = int(person_row["person_id"].values[0])

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        embedding = faces[0].embedding
        embeddings.append(embedding)
        id_mapping.append(person_id)

print("Total embeddings:", len(embeddings))

# -------- CONVERT TO NUMPY --------
embeddings = np.array(embeddings).astype("float32")

# -------- BUILD FAISS INDEX --------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -------- SAVE INDEX --------
faiss.write_index(index, INDEX_SAVE_PATH)

# -------- SAVE ID MAPPING --------
with open(MAPPING_SAVE_PATH, "wb") as f:
    pickle.dump(id_mapping, f)

print("FAISS index built and saved successfully!")