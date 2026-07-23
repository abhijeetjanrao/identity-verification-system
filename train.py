from src.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    result = pipeline.run_pipeline()
    print(result)
