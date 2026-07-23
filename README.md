# Identity Verification System

A full-stack AI project that demonstrates an end-to-end identity verification workflow using computer vision and deep learning. The system is designed for practical use cases such as attendance verification, access control demos, and person recognition.

## Project Overview

This project combines:
- a training pipeline for face embedding generation,
- a FastAPI backend for prediction and health monitoring,
- a Streamlit frontend for interactive image-based verification,
- and a simple history tracking flow for recent verification attempts.

## Key Features

- Face recognition and identity verification from uploaded images
- End-to-end training pipeline from dataset ingestion to FAISS indexing
- FastAPI-based inference API with health status endpoints
- Streamlit web app for easy demo use
- Graceful handling for missing model artifacts
- Docker support for deployment readiness
- Basic automated API tests

## Tech Stack

- Python
- FastAPI
- Streamlit
- OpenCV
- InsightFace
- FAISS
- NumPy / Pandas
- Docker

## Folder Structure

- app.py — FastAPI backend
- streamlit_app.py — Streamlit frontend
- train.py — entry point for training pipeline
- src/ — model training and prediction logic
- tests/ — basic API tests
- Dockerfile — container setup

## Getting Started

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train the model

```powershell
python train.py
```

### 4. Start the backend API

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Start the Streamlit UI

```powershell
streamlit run streamlit_app.py
```

### 6. Open the app

- API docs: http://127.0.0.1:8000/docs
- Streamlit UI: http://localhost:8501

## API Endpoints

- GET / — service status and readiness
- GET /health — health check
- GET /history — recent verification records
- POST /predict — upload an image and receive a prediction result

## Example Usage

```powershell
python client_predict.py C:\path\to\image.jpg
```

## Testing

```powershell
python -m unittest discover -s tests
```

## Notes

- Model artifacts are stored in the saved_models directory.
- If the model files are missing, the API returns a clear error instead of crashing.
- The project is structured to be easy to extend for attendance systems, access control demos, or personalized recognition workflows.

