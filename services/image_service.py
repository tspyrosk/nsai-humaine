"""
Image Service - Handles image dataset loading, tag extraction, and feature extraction.

This service converts image datasets into tabular format by:
1. Extracting tags from images using OpenAI Vision API
2. Extracting numeric features (GLCM texture, statistics) using Pillow/numpy
3. Creating a DataFrame with boolean tag columns + numeric image features
"""
import os
import re
import json
import base64
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as scipy_stats
from tqdm import tqdm

# OpenAI imports (optional - only needed for tag extraction)
try:
    from openai import OpenAI, RateLimitError, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from paths import INPUT_DIR


# =============================================================================
# Configuration
# =============================================================================

TAGS_CACHE_FILENAME = "tags_cache.json"
IMAGE_FEATURES_CACHE_FILENAME = "image_features_cache.json"
DATASET_HASH_FILENAME = ".dataset_hash"


# =============================================================================
# Image Loading
# =============================================================================

def load_image_dataset(dataset_path: str) -> Tuple[List[Path], List[int], List[str]]:
    """
    Load image paths and labels from a folder structure.
    Expects subfolders where each subfolder name is a class label.

    For binary classification, expects two subfolders. The folder names
    are used as class labels (e.g., "yes"/"no", "tumor"/"no_tumor", "1"/"0").

    Args:
        dataset_path: Path to the dataset folder

    Returns:
        Tuple of (image_paths, labels, class_names)
        - image_paths: List of Path objects to images
        - labels: List of integer labels (0 or 1 for binary)
        - class_names: List of class folder names
    """
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset folder not found: {dataset_path}")

    # Find class subfolders (skip hidden folders and cache files)
    class_folders = sorted([
        d for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if len(class_folders) < 2:
        raise ValueError(f"Expected at least 2 class subfolders, found {len(class_folders)}")

    class_names = [f.name for f in class_folders]

    # Determine which class is positive (1) and negative (0)
    # Heuristic: "yes", "1", "positive", "tumor" → positive class
    positive_indicators = {'yes', '1', 'positive', 'tumor', 'true', 'malignant'}

    positive_idx = None
    for i, name in enumerate(class_names):
        if name.lower() in positive_indicators:
            positive_idx = i
            break

    # If no heuristic match, use alphabetical order (first = 0, second = 1)
    if positive_idx is None:
        positive_idx = 1  # Second folder alphabetically is positive

    image_paths = []
    labels = []

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    for folder_idx, folder in enumerate(class_folders):
        label = 1 if folder_idx == positive_idx else 0

        for img_file in folder.iterdir():
            if img_file.suffix.lower() in image_extensions:
                image_paths.append(img_file)
                labels.append(label)

    return image_paths, labels, class_names


def get_sample_images(image_paths: List[Path], labels: List[int],
                      n_per_class: int = 4) -> Dict[int, List[Path]]:
    """
    Get sample images from each class for display.

    Args:
        image_paths: List of all image paths
        labels: List of corresponding labels
        n_per_class: Number of samples per class

    Returns:
        Dict mapping label to list of sample image paths
    """
    samples = {}
    for label in set(labels):
        class_images = [p for p, l in zip(image_paths, labels) if l == label]
        samples[label] = class_images[:n_per_class]
    return samples


# =============================================================================
# Dataset Hashing (for cache invalidation)
# =============================================================================

def compute_dataset_hash(dataset_path: str) -> str:
    """
    Compute a hash of the dataset based on file names and modification times.
    Used to detect when dataset changes and cache should be invalidated.
    """
    dataset_dir = Path(dataset_path)

    files_info = []
    for f in sorted(dataset_dir.rglob("*")):
        if f.is_file() and not f.name.startswith('.') and 'cache' not in f.name.lower():
            files_info.append(f"{f.relative_to(dataset_dir)}:{f.stat().st_mtime}")

    content = "\n".join(files_info)
    return hashlib.md5(content.encode()).hexdigest()


def is_cache_valid(dataset_path: str) -> bool:
    """
    Check if the cache is still valid for this dataset.
    """
    dataset_dir = Path(dataset_path)
    hash_file = dataset_dir / DATASET_HASH_FILENAME

    if not hash_file.exists():
        return False

    stored_hash = hash_file.read_text().strip()
    current_hash = compute_dataset_hash(dataset_path)

    return stored_hash == current_hash


def save_dataset_hash(dataset_path: str):
    """
    Save the current dataset hash for future cache validation.
    """
    dataset_dir = Path(dataset_path)
    hash_file = dataset_dir / DATASET_HASH_FILENAME
    current_hash = compute_dataset_hash(dataset_path)
    hash_file.write_text(current_hash)


# =============================================================================
# Tag Extraction (OpenAI Vision API)
# =============================================================================

def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_tags_from_image(client, image_path: Path, max_retries: int = 5) -> List[str]:
    """
    Use OpenAI Vision API to extract descriptive tags from an image.
    """
    if not OPENAI_AVAILABLE:
        return []

    base64_image = encode_image_to_base64(image_path)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image and provide a list of visual features and characteristics you observe.
Focus on features such as:
- Presence or absence of abnormal patterns
- Brightness/darkness patterns
- Symmetry or asymmetry
- Texture characteristics (smooth, rough, irregular)
- Contrast levels
- Shape characteristics
- Any visible abnormalities or notable features

Return ONLY a comma-separated list of descriptive tags, like: "bright_region, asymmetric, irregular_shape, high_contrast"
Use lowercase and underscores for multi-word tags."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            tags_str = response.choices[0].message.content.strip()
            tags = [tag.strip() for tag in tags_str.split(',')]
            time.sleep(0.5)
            return tags

        except RateLimitError:
            wait_time = (2 ** attempt) + (0.5 * attempt)
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                return []
        except APIError:
            wait_time = 2 * (attempt + 1)
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                return []
        except Exception:
            return []

    return []


def is_valid_tag(tag: str) -> bool:
    """
    Check if a tag is a valid feature tag (not a conversational response).
    """
    tag = tag.strip()

    if len(tag) > 50 or len(tag) == 0:
        return False

    conversational_starts = [
        "I ", "I'", "I'm", "Please", "However", "If you", "Let me",
        "feel free", "you can", "including", "based on"
    ]
    for pattern in conversational_starts:
        if tag.lower().startswith(pattern.lower()):
            return False

    conversational_markers = [
        "can help", "assist", "analyze", "interpret", "consult",
        "professional", "let me know", "questions", "description"
    ]
    tag_lower = tag.lower()
    for marker in conversational_markers:
        if marker in tag_lower:
            return False

    valid_pattern = re.compile(r'^[a-z][a-z0-9_]*$')
    if valid_pattern.match(tag):
        return True

    cleaned = tag.lower().replace('-', '_').replace('"', '').replace("'", '').strip()
    if valid_pattern.match(cleaned):
        return True

    return False


def filter_tags(tags: List[str]) -> List[str]:
    """Filter and normalize tags."""
    valid_tags = [tag for tag in tags if is_valid_tag(tag)]
    normalized_tags = [
        tag.lower().replace('-', '_').replace('"', '').strip()
        for tag in valid_tags
    ]
    return normalized_tags


def extract_and_cache_tags(dataset_path: str, image_paths: List[Path],
                           openai_api_key: Optional[str] = None,
                           force_refresh: bool = False,
                           progress_callback=None) -> Dict[str, List[str]]:
    """
    Extract tags for all images with caching.

    Args:
        dataset_path: Path to dataset folder (for cache storage)
        image_paths: List of image paths to process
        openai_api_key: OpenAI API key (optional, uses env var if not provided)
        force_refresh: If True, re-extract all tags
        progress_callback: Optional callback for progress updates

    Returns:
        Dict mapping image keys to tag lists
    """
    dataset_dir = Path(dataset_path)
    cache_file = dataset_dir / TAGS_CACHE_FILENAME

    # Load existing cache
    cache = {}
    if not force_refresh and cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)

    # Find images that need processing
    images_to_process = []
    for img_path in image_paths:
        img_key = str(img_path.relative_to(dataset_dir))
        if img_key not in cache:
            images_to_process.append((img_path, img_key))

    if not images_to_process:
        # Filter existing cache
        filtered_cache = {k: filter_tags(v) for k, v in cache.items()}
        return filtered_cache

    # Initialize OpenAI client
    if not OPENAI_AVAILABLE:
        # Return empty tags for images without OpenAI
        for img_path, img_key in images_to_process:
            cache[img_key] = []
    else:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            for img_path, img_key in images_to_process:
                cache[img_key] = []
        else:
            client = OpenAI(api_key=api_key)

            for i, (img_path, img_key) in enumerate(images_to_process):
                tags = extract_tags_from_image(client, img_path)
                cache[img_key] = tags

                # Save after each image
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2)

                if progress_callback:
                    progress_callback(i + 1, len(images_to_process))

                time.sleep(1.0)

    # Filter and return
    filtered_cache = {k: filter_tags(v) for k, v in cache.items()}
    return filtered_cache


# =============================================================================
# Image Feature Extraction (GLCM + Statistics)
# =============================================================================

def compute_glcm(img: np.ndarray, distance: int = 1, levels: int = 64) -> np.ndarray:
    """
    Compute Gray-Level Co-occurrence Matrix using numpy.
    """
    if levels < 256:
        img = (img / 256 * levels).astype(np.uint8)

    rows, cols = img.shape
    glcm = np.zeros((levels, levels, 4), dtype=np.float64)

    offsets = [(0, distance), (-distance, distance), (-distance, 0), (-distance, -distance)]

    for angle_idx, (dy, dx) in enumerate(offsets):
        for i in range(max(0, -dy), min(rows, rows - dy)):
            for j in range(max(0, -dx), min(cols, cols - dx)):
                row_val = img[i, j]
                col_val = img[i + dy, j + dx]
                glcm[row_val, col_val, angle_idx] += 1

    for angle_idx in range(4):
        total = glcm[:, :, angle_idx].sum()
        if total > 0:
            glcm[:, :, angle_idx] /= total

    return glcm


def glcm_props(glcm: np.ndarray) -> Dict[str, List[float]]:
    """
    Compute GLCM properties: contrast, homogeneity, energy, correlation.
    """
    levels = glcm.shape[0]
    n_angles = glcm.shape[2]

    i_idx, j_idx = np.meshgrid(range(levels), range(levels), indexing='ij')

    props = {'contrast': [], 'homogeneity': [], 'energy': [], 'correlation': []}

    for angle in range(n_angles):
        p = glcm[:, :, angle]

        contrast = np.sum((i_idx - j_idx) ** 2 * p)
        props['contrast'].append(float(contrast))

        homogeneity = np.sum(p / (1 + np.abs(i_idx - j_idx)))
        props['homogeneity'].append(float(homogeneity))

        energy = np.sum(p ** 2)
        props['energy'].append(float(energy))

        mu_i = np.sum(i_idx * p)
        mu_j = np.sum(j_idx * p)
        sigma_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * p))
        sigma_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * p))

        if sigma_i > 0 and sigma_j > 0:
            correlation = np.sum((i_idx - mu_i) * (j_idx - mu_j) * p) / (sigma_i * sigma_j)
        else:
            correlation = 0.0
        props['correlation'].append(float(correlation))

    return props


def extract_image_features(image_path: Path, target_size: Tuple[int, int] = (128, 128)) -> Dict[str, float]:
    """
    Extract CPU-friendly features from an image.

    Features:
    - Statistical: mean, std, skewness, kurtosis (4 features)
    - GLCM texture: contrast, homogeneity, energy, correlation at 4 angles (16 features)

    Total: 20 features per image
    """
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.uint8)

    features = {}

    # Statistical features
    img_normalized = img_array.astype(np.float64) / 255.0
    flat = img_normalized.flatten()
    features['stat_mean'] = float(np.mean(flat))
    features['stat_std'] = float(np.std(flat))
    features['stat_skewness'] = float(scipy_stats.skew(flat))
    features['stat_kurtosis'] = float(scipy_stats.kurtosis(flat))

    # GLCM features
    img_quantized = (img_array / 4).astype(np.uint8)
    glcm = compute_glcm(img_quantized, distance=1, levels=64)
    props = glcm_props(glcm)

    angle_names = ['0', '45', '90', '135']
    for i, angle_name in enumerate(angle_names):
        features[f'glcm_contrast_{angle_name}'] = props['contrast'][i]
        features[f'glcm_homogeneity_{angle_name}'] = props['homogeneity'][i]
        features[f'glcm_energy_{angle_name}'] = props['energy'][i]
        features[f'glcm_correlation_{angle_name}'] = props['correlation'][i]

    return features


def extract_and_cache_image_features(dataset_path: str, image_paths: List[Path],
                                      force_refresh: bool = False,
                                      progress_callback=None) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Extract image features with caching.

    Args:
        dataset_path: Path to dataset folder (for cache storage)
        image_paths: List of image paths to process
        force_refresh: If True, re-extract all features
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (features_cache, feature_names)
    """
    dataset_dir = Path(dataset_path)
    cache_file = dataset_dir / IMAGE_FEATURES_CACHE_FILENAME

    # Load existing cache
    cache = {}
    if not force_refresh and cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)

    # Find images that need processing
    images_to_process = []
    for img_path in image_paths:
        img_key = str(img_path.relative_to(dataset_dir))
        if img_key not in cache:
            images_to_process.append((img_path, img_key))

    if images_to_process:
        for i, (img_path, img_key) in enumerate(images_to_process):
            try:
                features = extract_image_features(img_path)
                cache[img_key] = features
            except Exception as e:
                cache[img_key] = {}

            if progress_callback:
                progress_callback(i + 1, len(images_to_process))

        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)

    # Get feature names
    feature_names = []
    for features in cache.values():
        if features:
            feature_names = sorted(features.keys())
            break

    return cache, feature_names


# =============================================================================
# Convert to Tabular Dataset
# =============================================================================

def create_tabular_dataset(dataset_path: str, image_paths: List[Path], labels: List[int],
                           tags_cache: Dict[str, List[str]],
                           image_features_cache: Dict[str, Dict[str, float]],
                           image_feature_names: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Convert image dataset to tabular format.

    Args:
        dataset_path: Path to dataset folder
        image_paths: List of image paths
        labels: List of labels
        tags_cache: Dict mapping image keys to tag lists
        image_features_cache: Dict mapping image keys to feature dicts
        image_feature_names: List of image feature names

    Returns:
        Tuple of (DataFrame, all_tags, all_feature_cols)
    """
    dataset_dir = Path(dataset_path)

    # Collect all unique tags
    all_tags = set()
    for tags in tags_cache.values():
        all_tags.update(tags)
    all_tags = sorted(list(all_tags))

    # Create feature matrix
    data = []
    for img_path, label in zip(image_paths, labels):
        img_key = str(img_path.relative_to(dataset_dir))
        img_tags = set(tags_cache.get(img_key, []))

        # Boolean features for tags
        features = {tag: (tag in img_tags) for tag in all_tags}

        # Numeric image features
        img_features = image_features_cache.get(img_key, {})
        for feat_name in image_feature_names:
            features[feat_name] = img_features.get(feat_name, 0.0)

        features['label'] = label
        features['image_path'] = img_key

        data.append(features)

    df = pd.DataFrame(data)

    # All feature columns (tags + image features)
    all_feature_cols = list(all_tags) + list(image_feature_names)

    return df, all_tags, all_feature_cols


# =============================================================================
# Main API Function
# =============================================================================

def load_image_dataset_as_tabular(dataset_path: str,
                                   openai_api_key: Optional[str] = None,
                                   force_refresh: bool = False,
                                   tag_progress_callback=None,
                                   feature_progress_callback=None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load an image dataset and convert it to tabular format.

    This is the main entry point for image dataset loading. It:
    1. Scans the dataset folder structure
    2. Extracts tags using OpenAI Vision API (with caching)
    3. Extracts image features (with caching)
    4. Creates a tabular DataFrame

    Args:
        dataset_path: Path to the dataset folder
        openai_api_key: OpenAI API key (optional, uses env var if not provided)
        force_refresh: If True, ignore cache and re-extract everything
        tag_progress_callback: Callback for tag extraction progress
        feature_progress_callback: Callback for feature extraction progress

    Returns:
        Tuple of (DataFrame, metadata_dict)
        - DataFrame: Tabular dataset with boolean tag columns, numeric features, and label
        - metadata_dict: Contains class_names, all_tags, image_feature_names, etc.
    """
    # Load image paths and labels
    image_paths, labels, class_names = load_image_dataset(dataset_path)

    # Check cache validity
    cache_valid = is_cache_valid(dataset_path) and not force_refresh

    # Extract tags
    tags_cache = extract_and_cache_tags(
        dataset_path, image_paths,
        openai_api_key=openai_api_key,
        force_refresh=not cache_valid,
        progress_callback=tag_progress_callback
    )

    # Extract image features
    image_features_cache, image_feature_names = extract_and_cache_image_features(
        dataset_path, image_paths,
        force_refresh=not cache_valid,
        progress_callback=feature_progress_callback
    )

    # Create tabular dataset
    df, all_tags, all_feature_cols = create_tabular_dataset(
        dataset_path, image_paths, labels,
        tags_cache, image_features_cache, image_feature_names
    )

    # Save dataset hash for future cache validation
    save_dataset_hash(dataset_path)

    metadata = {
        'class_names': class_names,
        'all_tags': all_tags,
        'image_feature_names': image_feature_names,
        'all_feature_cols': all_feature_cols,
        'n_images': len(image_paths),
        'n_positive': sum(labels),
        'n_negative': len(labels) - sum(labels),
    }

    return df, metadata


def list_available_image_datasets() -> List[str]:
    """
    List available image datasets in the input folder.

    Returns:
        List of dataset folder names that contain image subfolders
    """
    input_dir = Path(INPUT_DIR)
    datasets = []

    for folder in input_dir.iterdir():
        if folder.is_dir():
            # Check if it has at least 2 subfolders (class folders)
            subfolders = [d for d in folder.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if len(subfolders) >= 2:
                # Check if subfolders contain images
                has_images = False
                for sf in subfolders:
                    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
                    for f in sf.iterdir():
                        if f.suffix.lower() in image_extensions:
                            has_images = True
                            break
                    if has_images:
                        break

                if has_images:
                    datasets.append(folder.name)

    return datasets
