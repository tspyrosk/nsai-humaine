"""
Tag Standardization - Semantic tag clustering using sentence transformers.

Provides lightweight semantic similarity-based tag merging to reduce redundancy
in extracted tags (e.g., "abnormal_region" and "abnormal_area" -> "abnormal_area").
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import AgglomerativeClustering

# Lazy load sentence transformers to avoid import overhead when not needed
_st_model = None


def get_sentence_transformer_model():
    """Lazy load the sentence transformer model."""
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic tag standardization. "
                "Install with: pip install sentence-transformers"
            )
    return _st_model


def standardize_tags_semantic(
    tags: List[str],
    distance_threshold: float = 0.4
) -> Dict[str, str]:
    """
    Standardize tags by clustering semantically similar ones.

    Uses sentence-transformers (MiniLM-L6-v2) for semantic embeddings
    and agglomerative clustering to group similar tags.

    Args:
        tags: List of tag strings to standardize
        distance_threshold: Cosine distance threshold for clustering.
            Lower = more aggressive merging (0.3-0.4 recommended).
            Higher = more conservative (0.5-0.6).

    Returns:
        Dict mapping each original tag to its canonical (representative) tag.
        The canonical tag is the shortest in each cluster.

    Example:
        >>> tags = ["abnormal_region", "abnormal_area", "lesion", "mass"]
        >>> mapping = standardize_tags_semantic(tags, distance_threshold=0.4)
        >>> # {'abnormal_region': 'abnormal_area', 'abnormal_area': 'abnormal_area', ...}
    """
    if len(tags) < 2:
        return {t: t for t in tags}

    model = get_sentence_transformer_model()

    # Preprocess: replace underscores with spaces for better embedding
    processed = [t.replace('_', ' ') for t in tags]

    # Get embeddings
    embeddings = model.encode(processed)

    # Cluster similar tags
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # Map each tag to its canonical form (shortest in cluster)
    mapping = {}
    for cluster_id in set(labels):
        cluster_tags = [t for t, l in zip(tags, labels) if l == cluster_id]
        # Choose shortest tag as canonical (more concise)
        canonical = min(cluster_tags, key=len)
        for tag in cluster_tags:
            mapping[tag] = canonical

    return mapping


def standardize_tags_with_frequencies(
    tags: List[str],
    tag_frequencies: Dict[str, int],
    distance_threshold: float = 0.4
) -> Dict[str, str]:
    """
    Standardize tags, preferring more frequent tags as canonical.

    Similar to standardize_tags_semantic, but uses frequency to break ties
    when selecting the canonical tag for each cluster.

    Args:
        tags: List of tag strings to standardize
        tag_frequencies: Dict mapping tag to its occurrence count
        distance_threshold: Cosine distance threshold for clustering

    Returns:
        Dict mapping each original tag to its canonical tag.
        Canonical is selected by: highest frequency, then shortest length.
    """
    if len(tags) < 2:
        return {t: t for t in tags}

    model = get_sentence_transformer_model()

    # Preprocess
    processed = [t.replace('_', ' ') for t in tags]

    # Get embeddings
    embeddings = model.encode(processed)

    # Cluster
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # Map to canonical (most frequent, then shortest)
    mapping = {}
    for cluster_id in set(labels):
        cluster_tags = [t for t, l in zip(tags, labels) if l == cluster_id]
        # Sort by frequency (desc), then length (asc)
        canonical = max(cluster_tags, key=lambda t: (tag_frequencies.get(t, 0), -len(t)))
        for tag in cluster_tags:
            mapping[tag] = canonical

    return mapping


def apply_tag_mapping_to_cache(
    tags_cache: Dict[str, List[str]],
    tag_mapping: Dict[str, str]
) -> Dict[str, List[str]]:
    """
    Apply a tag mapping to a tags cache, replacing tags with their canonical forms.

    Args:
        tags_cache: Dict mapping keys (e.g., image paths) to tag lists
        tag_mapping: Dict mapping original tags to canonical tags

    Returns:
        New cache with tags replaced by canonical forms (duplicates removed)
    """
    merged_cache = {}
    for key, tags in tags_cache.items():
        # Map tags to canonical forms and remove duplicates
        merged_tags = list(set(tag_mapping.get(tag, tag) for tag in tags))
        merged_cache[key] = merged_tags
    return merged_cache


def get_merged_clusters(tag_mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Get clusters showing which tags were merged together.

    Args:
        tag_mapping: Dict mapping original tags to canonical tags

    Returns:
        Dict mapping canonical tag to list of original tags that were merged into it
        (only includes clusters where merging occurred)
    """
    clusters = {}
    for original, canonical in tag_mapping.items():
        if canonical not in clusters:
            clusters[canonical] = []
        if original != canonical:
            clusters[canonical].append(original)

    # Only return clusters with merged tags
    return {k: v for k, v in clusters.items() if v}


def compute_standardization_stats(
    original_tags: List[str],
    tag_mapping: Dict[str, str]
) -> Dict:
    """
    Compute statistics about the tag standardization.

    Args:
        original_tags: List of original unique tags
        tag_mapping: Mapping from original to canonical tags

    Returns:
        Dict with statistics about the standardization
    """
    canonical_tags = set(tag_mapping.values())
    merged_clusters = get_merged_clusters(tag_mapping)

    return {
        'original_count': len(original_tags),
        'canonical_count': len(canonical_tags),
        'reduction': len(original_tags) - len(canonical_tags),
        'reduction_percent': (len(original_tags) - len(canonical_tags)) / max(len(original_tags), 1) * 100,
        'merged_groups': len(merged_clusters),
        'merged_clusters_sample': dict(list(merged_clusters.items())[:5])
    }
