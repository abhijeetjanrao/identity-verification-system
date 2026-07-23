from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataIngestionConfig
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def __init__(self):
        self.artifact_dir = "saved_models"   # ✅ Single artifact folder


    def start_data_ingestion(self):

        config = DataIngestionConfig(
            dataset_path="bollywood_celeb_faces_0",
            metadata_path="metadata.csv",
            ingestion_artifact_dir="artifacts/data_ingestion"
        )

        ingestion = DataIngestion(config)
        dataset_path, metadata_path = ingestion.initiate_data_ingestion()

        return dataset_path, metadata_path


    def start_data_transformation(self, dataset_path, metadata_path):

        transformer = DataTransformation(
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            artifact_dir=self.artifact_dir   # ✅ SAME folder
        )

        embeddings, labels = transformer.initiate_transformation()

        return embeddings, labels


    def run_pipeline(self):

        dataset_path, metadata_path = self.start_data_ingestion()

        embeddings, labels = self.start_data_transformation(
            dataset_path, metadata_path
        )

        print("Pipeline completed successfully.")

        trainer = ModelTrainer(
            artifact_dir=self.artifact_dir   # ✅ SAME folder
        )

        trainer.initiate_model_training()

        return {
            "dataset_path": dataset_path,
            "metadata_path": metadata_path,
            "artifact_dir": self.artifact_dir,
            "status": "completed",
        }