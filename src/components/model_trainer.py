import os
import numpy as np
import faiss


class ModelTrainer:

    def __init__(self, artifact_dir):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def initiate_model_training(self):

        print("Loading embeddings...")

        embeddings = np.load(os.path.join(self.artifact_dir, "embeddings.npy"))
        labels = np.load(os.path.join(self.artifact_dir, "labels.npy"))

        print("Embeddings shape:", embeddings.shape)

        dimension = embeddings.shape[1]

        # L2 Normalization (important for cosine similarity)
        faiss.normalize_L2(embeddings)

        # Create FAISS Index
        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings)

        # Save FAISS index
        faiss.write_index(
            index,
            os.path.join(self.artifact_dir, "faiss_index.bin")
        )

        print("Model training completed.")
        print("Total faces indexed:", index.ntotal)

        return index