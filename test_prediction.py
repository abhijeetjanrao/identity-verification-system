from src.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

result = pipeline.predict("test.jpg")

print(result)