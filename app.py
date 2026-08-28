import os
import shutil
import tempfile
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI(title="Identity Verification API")

predictor = None
history_records = []


def get_predictor():
    global predictor

    if predictor is None:
        from src.pipeline.prediction_pipeline import PredictionPipeline
        predictor = PredictionPipeline()

    return predictor


# Load the AI model when the server starts
@app.on_event("startup")
def startup_event():
    get_predictor()


@app.get("/")
def home():
    try:
        model = get_predictor()

        return {
            "status": "API Running Successfully",
            "service": "Identity Verification System",
            "model_ready": model.ready,
            "message": (
                "Model ready"
                if model.ready
                else (model.error or "Model not ready")
            ),
        }

    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "API Running",
                "service": "Identity Verification System",
                "model_ready": False,
                "error": str(exc),
            },
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Identity Verification System",
    }


@app.get("/history")
def get_history():
    return {
        "records": history_records
    }


@app.post("/predict")
async def predict_face(file: UploadFile = File(...)):

    if file.filename is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "A file is required"
            },
        )

    _, ext = os.path.splitext(file.filename.lower())

    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unsupported image format"
            },
        )

    temp_path = None

    try:
        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"{uuid4().hex}{ext}",
        )

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        model = get_predictor()

        if not model.ready:
            return JSONResponse(
                status_code=503,
                content={
                    "error": model.error or "Model not ready",
                    "status": "unavailable",
                },
            )

        result = model.predict(temp_path)

        history_records.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "result": result,
            }
        )

        return result

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Prediction failed: {exc}"
            },
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
