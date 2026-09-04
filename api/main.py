"""FastAPI backend for Sign Language to Text inference."""

import numpy as np
import pickle
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Load model and encoder once at startup ---
model = tf.keras.models.load_model("model/test_model_clustered.h5")
with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

app = FastAPI(title="Sign Language to Text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict for production
    allow_methods=["*"],
    allow_headers=["*"],
)


class LandmarkRequest(BaseModel):
    landmarks: list[float]


class PredictionResponse(BaseModel):
    letter: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(body: LandmarkRequest):
    if len(body.landmarks) != 63:
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly 63 landmark values, got {len(body.landmarks)}.",
        )

    input_array = np.array([body.landmarks])
    prediction = model.predict(input_array, verbose=0)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    letter = encoder.inverse_transform([predicted_class])[0]

    return PredictionResponse(letter=letter, confidence=confidence)
