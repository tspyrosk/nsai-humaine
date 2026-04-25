"""
NSAI-Humaine Inference API — embedding-only.

The service is stateless: callers supply the tag vocabulary with every request.
The expected flow is that a consumer (e.g. the integration notebooks) reads
feature_names.txt from the latest published model, splits it into tag vs
numeric columns, and passes the tag list in on each /embed/* call.

Endpoints:
  POST /embed/text   {text, tags}       -> {features ordered to match tags}
  POST /embed/image  image + tags form  -> {tag_features, numeric_features}

Scaling and model prediction are the consumer's responsibility.

Port: 8888 (configurable via PORT env var).
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PORT = int(os.getenv("PORT", "8888"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_extractor = None


def _load_resources() -> None:
    global _extractor
    from inference.feature_extractor import InferenceFeatureExtractor
    _extractor = InferenceFeatureExtractor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    log.info("Feature extractor ready (stateless, tags supplied per request).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_resources()
    yield


app = FastAPI(
    title="NSAI-Humaine Inference API",
    description=(
        "Stateless embedding service. Callers provide the tag vocabulary "
        "(read from their published model's feature_names.txt) on every request."
    ),
    version="2.0.0",
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

class TextEmbedRequest(BaseModel):
    text: str
    tags: List[str] = Field(..., description="Tag vocabulary to score, in the order the caller expects them back.")


class TextEmbedResponse(BaseModel):
    features: List[int]
    tags: List[str]
    n_features: int


class ImageEmbedResponse(BaseModel):
    tag_features: List[int]
    tags: List[str]
    numeric_features: Dict[str, float]


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok" if _extractor is not None else "degraded"}


# ------------------------------------------------------------------
# Embed endpoints
# ------------------------------------------------------------------

@app.post("/embed/text", response_model=TextEmbedResponse)
def embed_text(req: TextEmbedRequest):
    """
    Score each supplied tag as 0 or 1 for the given text.
    The returned `features` list is ordered to match `tags` in the request.
    """
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Extractor not initialised.")
    features = _extractor.embed_text(req.text, req.tags)
    return TextEmbedResponse(features=features, tags=req.tags, n_features=len(features))


@app.post("/embed/image", response_model=ImageEmbedResponse)
async def embed_image(
    image: UploadFile = File(...),
    tags: str = Form(..., description="JSON-encoded list of tag names."),
):
    """
    Score each supplied tag as 0 or 1, and return 20 deterministic GLCM +
    statistical numeric features (keyed by canonical name).

    `tags` is a JSON-encoded list string (multipart forms can't carry arrays
    natively), e.g. tags='["bright_region","symmetric"]'.
    """
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Extractor not initialised.")
    try:
        tag_list = json.loads(tags)
        if not isinstance(tag_list, list) or not all(isinstance(t, str) for t in tag_list):
            raise ValueError
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(
            status_code=422,
            detail="`tags` must be a JSON-encoded list of strings.",
        )

    image_bytes = await image.read()
    tag_features, numeric_features = _extractor.embed_image(image_bytes, tag_list)
    return ImageEmbedResponse(
        tag_features=tag_features,
        tags=tag_list,
        numeric_features=numeric_features,
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
