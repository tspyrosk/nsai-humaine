"""
InferenceFeatureExtractor — stateless.

Tags are supplied per request by the caller (typically an integration notebook
that has just read feature_names.txt from the published model's output dir).
No training artifacts are loaded on the server.

For text:  LLM scores each supplied tag as 0 or 1.
For image: 20 deterministic GLCM + statistical features are computed locally,
           and the LLM scores each supplied tag as 0 or 1.

Scaling (scaler.pkl) is intentionally left to the caller.
"""
import base64
import io
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class InferenceFeatureExtractor:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    def embed_text(self, text: str, tags: List[str]) -> List[int]:
        """Return 0/1 scores ordered to match `tags`."""
        scores = self._constrained_text_extraction(text, tags)
        return [scores.get(t, 0) for t in tags]

    def embed_image(
        self, image_bytes: bytes, tags: List[str]
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Return (tag_features ordered to match `tags`, numeric_features dict).
        The numeric features are the 20 deterministic GLCM + statistical values
        keyed by their canonical names (stat_mean, glcm_contrast_0, ...).
        """
        numeric = _compute_image_numeric_features(image_bytes)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        scores = self._constrained_image_extraction(b64, tags)
        return [scores.get(t, 0) for t in tags], numeric

    # ------------------------------------------------------------------
    # Constrained LLM extraction
    # ------------------------------------------------------------------

    def _constrained_text_extraction(
        self, text: str, tags: List[str]
    ) -> Dict[str, int]:
        if not tags:
            return {}
        if not OPENAI_AVAILABLE or not self.openai_api_key:
            return {t: 0 for t in tags}

        client = OpenAI(api_key=self.openai_api_key)
        prompt = (
            "You are classifying text according to a fixed set of semantic features.\n\n"
            f"Tags (the ONLY tags you must score, do not add or remove any): "
            f"{json.dumps(tags)}\n\n"
            f'Text:\n"""{text[:1500]}"""\n\n'
            "For each tag output 1 if the concept is clearly present in the text, "
            "0 if absent or unclear.\n"
            "Respond with ONLY a JSON object mapping every tag exactly as written to 0 or 1."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            result = json.loads(response.choices[0].message.content)
            return {t: int(bool(result.get(t, 0))) for t in tags}
        except Exception:
            return {t: 0 for t in tags}

    def _constrained_image_extraction(
        self, b64_image: str, tags: List[str]
    ) -> Dict[str, int]:
        if not tags:
            return {}
        if not OPENAI_AVAILABLE or not self.openai_api_key:
            return {t: 0 for t in tags}

        client = OpenAI(api_key=self.openai_api_key)
        prompt = (
            "You are analyzing an image according to a fixed set of visual features.\n\n"
            f"Tags (the ONLY tags you must score, do not add or remove any): "
            f"{json.dumps(tags)}\n\n"
            "For each tag output 1 if the visual feature is clearly present in the image, "
            "0 if absent or unclear.\n"
            "Respond with ONLY a JSON object mapping every tag exactly as written to 0 or 1."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                }],
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            result = json.loads(response.choices[0].message.content)
            return {t: int(bool(result.get(t, 0))) for t in tags}
        except Exception:
            return {t: 0 for t in tags}


# ------------------------------------------------------------------
# Deterministic image feature computation
#
# Self-contained copy of the GLCM logic from image_service.py so the
# inference container has no dependency on the Streamlit app code.
# ------------------------------------------------------------------

def _compute_image_numeric_features(image_bytes: bytes) -> Dict[str, float]:
    """
    20 deterministic features: 4 statistical + 16 GLCM (4 properties × 4 angles).
    """
    from PIL import Image
    from scipy import stats as scipy_stats

    img = (
        Image.open(io.BytesIO(image_bytes))
        .convert("L")
        .resize((128, 128), Image.Resampling.LANCZOS)
    )
    img_array = np.array(img, dtype=np.uint8)

    features: Dict[str, float] = {}

    flat = (img_array.astype(np.float64) / 255.0).flatten()
    features["stat_mean"] = float(np.mean(flat))
    features["stat_std"] = float(np.std(flat))
    features["stat_skewness"] = float(scipy_stats.skew(flat))
    features["stat_kurtosis"] = float(scipy_stats.kurtosis(flat))

    glcm = _compute_glcm((img_array / 4).astype(np.uint8), distance=1, levels=64)
    props = _glcm_props(glcm)
    for i, angle in enumerate(["0", "45", "90", "135"]):
        features[f"glcm_contrast_{angle}"] = props["contrast"][i]
        features[f"glcm_homogeneity_{angle}"] = props["homogeneity"][i]
        features[f"glcm_energy_{angle}"] = props["energy"][i]
        features[f"glcm_correlation_{angle}"] = props["correlation"][i]

    return features


def _compute_glcm(img: np.ndarray, distance: int = 1, levels: int = 64) -> np.ndarray:
    if levels < 256:
        img = (img / 256 * levels).astype(np.uint8)
    rows, cols = img.shape
    glcm = np.zeros((levels, levels, 4), dtype=np.float64)
    offsets = [(0, distance), (-distance, distance), (-distance, 0), (-distance, -distance)]
    for angle_idx, (dy, dx) in enumerate(offsets):
        for i in range(max(0, -dy), min(rows, rows - dy)):
            for j in range(max(0, -dx), min(cols, cols - dx)):
                glcm[img[i, j], img[i + dy, j + dx], angle_idx] += 1
    for angle_idx in range(4):
        total = glcm[:, :, angle_idx].sum()
        if total > 0:
            glcm[:, :, angle_idx] /= total
    return glcm


def _glcm_props(glcm: np.ndarray) -> Dict[str, list]:
    levels, _, n_angles = glcm.shape
    i_idx, j_idx = np.meshgrid(range(levels), range(levels), indexing="ij")
    props: Dict[str, list] = {
        "contrast": [], "homogeneity": [], "energy": [], "correlation": []
    }
    for a in range(n_angles):
        p = glcm[:, :, a]
        props["contrast"].append(float(np.sum((i_idx - j_idx) ** 2 * p)))
        props["homogeneity"].append(float(np.sum(p / (1 + np.abs(i_idx - j_idx)))))
        props["energy"].append(float(np.sum(p ** 2)))
        mu_i = np.sum(i_idx * p)
        mu_j = np.sum(j_idx * p)
        sig_i = float(np.sqrt(np.sum((i_idx - mu_i) ** 2 * p)))
        sig_j = float(np.sqrt(np.sum((j_idx - mu_j) ** 2 * p)))
        corr = (
            float(np.sum((i_idx - mu_i) * (j_idx - mu_j) * p) / (sig_i * sig_j))
            if sig_i > 0 and sig_j > 0
            else 0.0
        )
        props["correlation"].append(corr)
    return props
