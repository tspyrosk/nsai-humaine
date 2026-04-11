"""
NSAI-Humaine Inference API

Exposes /embed and /predict endpoints for text, image, and tabular data.
Features returned by /embed are guaranteed to match the training feature space:
  - Same columns, same order  (feature_names.txt)
  - Same vocabulary           (tag_vocabulary.json)
  - Same scaling              (scaler.pkl)

Port: 8888 (configurable via PORT env var)
Output dir: /app/output (configurable via OUTPUT_DIR env var)
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
PORT = int(os.getenv("PORT", "8888"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Module-level singletons populated at startup
_extractor = None
_model = None


def _load_resources() -> None:
    global _extractor, _model

    from inference.feature_extractor import InferenceFeatureExtractor

    try:
        _extractor = InferenceFeatureExtractor(
            output_dir=OUTPUT_DIR,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        log.info(
            "Feature extractor loaded — data_type=%s, n_features=%d",
            _extractor.data_type,
            len(_extractor.feature_names),
        )
    except FileNotFoundError as exc:
        log.warning("Feature extractor not ready: %s", exc)
        _extractor = None

    model_path = os.path.join(OUTPUT_DIR, "ltn.h5")
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(model_path)
            log.info("Model loaded from %s", model_path)
        except Exception as exc:
            log.warning("Model load failed: %s", exc)
            _model = None
    else:
        log.warning("No model found at %s", model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_resources()
    yield


app = FastAPI(
    title="NSAI-Humaine Inference API",
    description=(
        "Embed text or images into the exact feature space used during training, "
        "then optionally run the trained LTN model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class TextRequest(BaseModel):
    text: str


class TabularRequest(BaseModel):
    row: Dict[str, float]


class EmbedResponse(BaseModel):
    features: List[float]
    feature_names: List[str]
    data_type: str
    n_features: int


class PredictResponse(BaseModel):
    score: float
    label: int
    features: List[float]
    feature_names: List[str]
    data_type: str


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _require_extractor():
    if _extractor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Feature extractor not ready. "
                "Run the training pipeline to generate tag_vocabulary.json, "
                "feature_names.txt, and scaler.pkl in the output directory."
            ),
        )


def _run_model(vector: np.ndarray) -> Tuple[float, int]:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Train a model first.",
        )
    raw = _model.predict(vector.reshape(1, -1), verbose=0)
    score = float(np.squeeze(raw))
    return score, int(score > 0.5)


def _to_embed_response(vector: np.ndarray, names: List[str], data_type: str) -> EmbedResponse:
    return EmbedResponse(
        features=vector.tolist(),
        feature_names=names,
        data_type=data_type,
        n_features=len(names),
    )


def _to_predict_response(
    vector: np.ndarray, names: List[str], data_type: str
) -> PredictResponse:
    score, label = _run_model(vector)
    return PredictResponse(
        score=score,
        label=label,
        features=vector.tolist(),
        feature_names=names,
        data_type=data_type,
    )


# ------------------------------------------------------------------
# Health / metadata
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if _extractor is not None else "degraded",
        "extractor_ready": _extractor is not None,
        "model_ready": _model is not None,
        "data_type": _extractor.data_type if _extractor else None,
        "n_features": len(_extractor.feature_names) if _extractor else None,
    }


@app.get("/features")
def list_features():
    """
    Return the ordered feature names from the training run.
    Consumers can use this to verify that an embedding vector they received
    maps to the correct columns.
    """
    _require_extractor()
    return {
        "feature_names": _extractor.feature_names,
        "tags": _extractor.tags,
        "image_feature_names": _extractor.image_feature_names,
        "data_type": _extractor.data_type,
        "n_features": len(_extractor.feature_names),
    }


# ------------------------------------------------------------------
# Embed endpoints — return the feature vector, no model call
# ------------------------------------------------------------------

@app.post("/embed/text", response_model=EmbedResponse)
def embed_text(req: TextRequest):
    """
    Map raw text to the training feature vector.

    Tags in the response correspond exactly to the vocabulary selected during
    training. No new tags are introduced at inference time.
    """
    _require_extractor()
    vector, names = _extractor.embed_text(req.text)
    return _to_embed_response(vector, names, "text")


@app.post("/embed/image", response_model=EmbedResponse)
async def embed_image(image: UploadFile = File(...)):
    """
    Map a raw image to the training feature vector.

    GLCM features are computed deterministically. Visual tags are constrained
    to the vocabulary selected during training.
    """
    _require_extractor()
    image_bytes = await image.read()
    vector, names = _extractor.embed_image(image_bytes)
    return _to_embed_response(vector, names, "image")


@app.post("/embed/tabular", response_model=EmbedResponse)
def embed_tabular(req: TabularRequest):
    """
    Map a {column: value} dict to the training feature vector.
    Column names must match those in feature_names.txt.
    """
    _require_extractor()
    vector, names = _extractor.embed_tabular(req.row)
    return _to_embed_response(vector, names, "tabular")


# ------------------------------------------------------------------
# Predict endpoints — embed then run the model
# ------------------------------------------------------------------

@app.post("/predict/text", response_model=PredictResponse)
def predict_text(req: TextRequest):
    """Embed text and run the trained LTN model."""
    _require_extractor()
    vector, names = _extractor.embed_text(req.text)
    return _to_predict_response(vector, names, "text")


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(image: UploadFile = File(...)):
    """Embed an image and run the trained LTN model."""
    _require_extractor()
    image_bytes = await image.read()
    vector, names = _extractor.embed_image(image_bytes)
    return _to_predict_response(vector, names, "image")


@app.post("/predict/tabular", response_model=PredictResponse)
def predict_tabular(req: TabularRequest):
    """Embed a tabular row and run the trained LTN model."""
    _require_extractor()
    vector, names = _extractor.embed_tabular(req.row)
    return _to_predict_response(vector, names, "tabular")


# ------------------------------------------------------------------
# Entry point (used when running directly, not via uvicorn CLI)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
