import os
import shutil
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):

        os.makedirs(self.config.ingestion_artifact_dir, exist_ok=True)

        # Copy dataset
        dataset_dest = os.path.join(self.config.ingestion_artifact_dir, "dataset")
        metadata_dest = os.path.join(self.config.ingestion_artifact_dir, "metadata.csv")

        shutil.copytree(self.config.dataset_path, dataset_dest, dirs_exist_ok=True)
        shutil.copy(self.config.metadata_path, metadata_dest)

        print("Data ingestion completed.")

        return dataset_dest, metadata_dest