```python
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


# ---------------------------------------------------------
# Root endpoint
# Lightweight endpoint - DOES NOT load the AI model
# ---------------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "API Running Successfully",
        "service": "Identity Verification System",
    }


# ---------------------------------------------------------
# Health check
# Lightweight endpoint - DOES NOT load the AI model
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Identity Verification System",
    }


# ---------------------------------------------------------
# History
# ---------------------------------------------------------
@app.get("/history")
def get_history():
    return {
        "records": history_records
    }


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
@app.post("/predict")
async def predict_face(file: UploadFile = File(...)):

    if file.filename is None:
        return JSONResponse(
            status_code=400,
            content={"error": "A file is required"},
        )

    _, ext = os.path.splitext(file.filename.lower())

    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported image format"},
        )

    temp_path = None

    try:
        # -------------------------------------------------
        # Create temporary image file
        # -------------------------------------------------
        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"{uuid4().hex}{ext}",
        )

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # -------------------------------------------------
        # Load prediction model
        # -------------------------------------------------
        model = get_predictor()

        if not model.ready:
            return JSONResponse(
                status_code=503,
                content={
                    "error": model.error or "Model not ready",
                    "status": "unavailable",
                },
            )

        # -------------------------------------------------
        # Run prediction
        # -------------------------------------------------
        result = model.predict(temp_path)

        # -------------------------------------------------
        # Save history
        # -------------------------------------------------
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
        # -------------------------------------------------
        # Remove temporary image
        # -------------------------------------------------
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
```
