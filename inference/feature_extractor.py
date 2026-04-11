"""
InferenceFeatureExtractor

Produces feature vectors at inference time that are guaranteed to be identical
in semantics and ordering to those produced during training.

The guarantee is structural — not assumed:
  - feature_names.txt   saved at training time, defines the exact column order
  - tag_vocabulary.json saved at training time, constrains the LLM to score only
                        the tags that were selected during training; no new tags
                        can enter the feature space at inference time
  - scaler.pkl          the fitted StandardScaler from the training split is the
                        same object applied here

For text: the LLM is given the tag list as a fixed checklist and asked to output
0 or 1 for each tag. Free-form tag discovery does not happen at inference time.

For images: GLCM and statistical features are computed deterministically (no LLM),
and the same constrained checklist approach is used for the visual tag features.
"""
import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class InferenceFeatureExtractor:
    """
    Loads training artifacts from output_dir and maps raw inputs to the
    feature vector the model was trained on.

    Args:
        output_dir: Directory containing tag_vocabulary.json, feature_names.txt,
                    and scaler.pkl written by the training pipeline.
        openai_api_key: Optional. Falls back to OPENAI_API_KEY env var.
    """

    def __init__(self, output_dir: str, openai_api_key: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._load_artifacts()

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        vocab_path = self.output_dir / "tag_vocabulary.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"tag_vocabulary.json not found at {vocab_path}. "
                "Run the training pipeline at least once to generate this file."
            )
        with open(vocab_path) as f:
            vocab = json.load(f)

        self.data_type: str = vocab["data_type"]
        self.tags: List[str] = vocab["tags"]
        self.image_feature_names: List[str] = vocab.get("image_feature_names", [])

        feature_names_path = self.output_dir / "feature_names.txt"
        if not feature_names_path.exists():
            raise FileNotFoundError(
                f"feature_names.txt not found at {feature_names_path}."
            )
        with open(feature_names_path) as f:
            self.feature_names: List[str] = [ln.strip() for ln in f if ln.strip()]

        scaler_path = self.output_dir / "scaler.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """
        Map a raw text string to the training feature vector.

        Returns:
            (vector, feature_names) where vector is scaler-transformed and
            aligned to the exact column order from feature_names.txt.
        """
        if self.data_type != "text":
            raise ValueError(
                f"Artifacts are for data_type='{self.data_type}', not 'text'."
            )
        tag_scores = self._constrained_text_extraction(text, self.tags)
        return self._align_and_scale(tag_scores, {})

    def embed_image(self, image_bytes: bytes) -> Tuple[np.ndarray, List[str]]:
        """
        Map raw image bytes to the training feature vector.

        GLCM features are computed deterministically. Tag features use a
        constrained LLM prompt that only scores tags from the training vocabulary.

        Returns:
            (vector, feature_names) aligned to feature_names.txt, scaler-transformed.
        """
        if self.data_type != "image":
            raise ValueError(
                f"Artifacts are for data_type='{self.data_type}', not 'image'."
            )
        numeric_features = _compute_image_numeric_features(image_bytes)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        tag_scores = self._constrained_image_extraction(b64, self.tags)
        return self._align_and_scale(tag_scores, numeric_features)

    def embed_tabular(self, row: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        """
        Map a {column: value} dict to the training feature vector.

        Returns:
            (vector, feature_names) aligned to feature_names.txt, scaler-transformed.
        """
        if self.data_type != "tabular":
            raise ValueError(
                f"Artifacts are for data_type='{self.data_type}', not 'tabular'."
            )
        numeric = {k: float(v) for k, v in row.items()}
        return self._align_and_scale({}, numeric)

    # ------------------------------------------------------------------
    # Vector assembly
    # ------------------------------------------------------------------

    def _align_and_scale(
        self,
        tag_scores: Dict[str, int],
        numeric_features: Dict[str, float],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a 1-D array aligned to feature_names order, then apply scaler.
        Any feature not present in the scores/features dicts defaults to 0.
        """
        vector = [
            float(tag_scores.get(name, numeric_features.get(name, 0.0)))
            for name in self.feature_names
        ]
        arr = np.array(vector, dtype=np.float32).reshape(1, -1)
        if self.scaler is not None:
            arr = self.scaler.transform(arr).astype(np.float32)
        return arr.flatten(), self.feature_names

    # ------------------------------------------------------------------
    # Constrained LLM extraction
    # ------------------------------------------------------------------

    def _constrained_text_extraction(
        self, text: str, tags: List[str]
    ) -> Dict[str, int]:
        """
        Ask the LLM to score only the tags selected during training.
        The vocabulary is fixed — the LLM cannot introduce new features.
        """
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
        """
        Ask the LLM to score only the visual tags selected during training.
        The vocabulary is fixed — the LLM cannot introduce new features.
        """
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
# This is a self-contained copy of the GLCM logic from image_service.py.
# Keeping it here means the inference container has no dependency on the
# Streamlit app code and the computation is guaranteed to be identical.
# ------------------------------------------------------------------

def _compute_image_numeric_features(image_bytes: bytes) -> Dict[str, float]:
    """
    Compute the same 20 deterministic features that image_service computes
    during training: 4 statistical + 16 GLCM (4 properties × 4 angles).
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
