from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    dataset_path: str
    metadata_path: str
    ingestion_artifact_dir: str