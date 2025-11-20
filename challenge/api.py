from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import cv2
import numpy as np
import fastapi
from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel

from ultralytics import YOLO

# Path to the trained model
MODEL_PATH = Path(__file__).parent / "artifacts" / "model_best.pt"

# Global model instance - Load on module import
model: Optional[YOLO] = None

def load_model_if_needed():
    """Load model if not already loaded"""
    global model
    if model is None and MODEL_PATH.exists():
        try:
            model = YOLO(str(MODEL_PATH))
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            model = None

# Load model on module import (for TestClient compatibility)
load_model_if_needed()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Ensure model is loaded (for production server)
    load_model_if_needed()

    yield

    # Shutdown: cleanup if needed
    pass


app = fastapi.FastAPI(lifespan=lifespan)


class BBox(BaseModel):
    """Bounding box in xyxy format (absolute pixel coordinates)"""
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    """Single object detection"""
    cls_id: int
    bbox: BBox


class PredictionResponse(BaseModel):
    """Response format for /predict endpoint"""
    detections: List[Detection]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Health check endpoint"""
    if model is None:
        return {
            "status": "model_not_loaded",
        }
    return {
        "status": "model_loaded",
    }


@app.post("/predict", status_code=200, response_model=PredictionResponse)
async def post_predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Run object detection on uploaded image

    Args:
        file: Image file (JPEG, PNG, etc.)

    Returns:
        JSON with detections in format:
        {
            "detections": [
                {
                    "cls_id": int,
                    "bbox": {"x1": int, "y1": int, "x2": int, "y2": int}
                }
            ]
        }
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # Run inference
        # Use conf=0.25 as default, can be adjusted for better recall
        results = model.predict(source=img, conf=0.25, verbose=False)

        # Convert results to expected format
        detections = []

        if len(results) > 0:
            result = results[0]  # First (and only) image

            # Extract boxes and classes
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # xyxy format
                classes = result.boxes.cls.cpu().numpy()  # class IDs

                for box, cls_id in zip(boxes, classes):
                    x1, y1, x2, y2 = box

                    detection = Detection(
                        cls_id=int(cls_id),
                        bbox=BBox(
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2)
                        )
                    )
                    detections.append(detection)

        return PredictionResponse(detections=detections)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
