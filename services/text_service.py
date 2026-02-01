"""
Text Service - Handles text dataset loading, tag extraction, and feature extraction.

This service converts text datasets into tabular format by:
1. Extracting semantic tags from text using OpenAI API (batched for efficiency)
2. Creating a DataFrame with boolean tag columns for each extracted tag
"""
import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# OpenAI imports (optional - only needed for tag extraction)
try:
    from openai import OpenAI, RateLimitError, APIError
    from pydantic import BaseModel, Field
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class QuotaExhaustedError(Exception):
    """Raised when OpenAI API quota is exhausted."""
    pass


def is_quota_error(error) -> bool:
    """Check if an API error is due to quota exhaustion (not transient rate limit)."""
    error_str = str(error).lower()
    quota_indicators = ['quota', 'exceeded', 'billing', 'insufficient_quota', 'budget']
    return any(indicator in error_str for indicator in quota_indicators)

from paths import INPUT_DIR

# Import tag standardization utilities
from services.tag_standardization import (
    standardize_tags_with_frequencies,
    apply_tag_mapping_to_cache,
    compute_standardization_stats,
    get_merged_clusters
)


# =============================================================================
# Configuration
# =============================================================================

TAGS_CACHE_FILENAME = "text_tags_cache.json"
DATASET_HASH_FILENAME = ".text_dataset_hash"
DEFAULT_BATCH_SIZE = 150
DEFAULT_DELAY_BETWEEN_BATCHES = 1.0
# Semantic similarity threshold for tag merging (lower = more aggressive)
TAG_SIMILARITY_THRESHOLD = 0.4

# Reserved column names that should not be used as tags
RESERVED_COLUMNS = {'text', 'message', 'label', 'index', 'id'}


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

if OPENAI_AVAILABLE:
    class MessageTags(BaseModel):
        """Tags extracted from a single text message."""
        tags: List[str] = Field(description="List of descriptive tags for this message")

    class BatchMessageTags(BaseModel):
        """Batch response containing tags for multiple text messages."""
        results: List[MessageTags] = Field(description="List of tags for each message in the batch, in order")


# =============================================================================
# Text Loading
# =============================================================================

def load_text_dataset(file_path: str,
                      text_column: Optional[str] = None,
                      label_column: Optional[str] = None,
                      delimiter: str = None) -> Tuple[List[str], List[int], List[str]]:
    """
    Load text data and labels from a file.

    Supports:
    - CSV files with headers
    - TSV files (like SMS Spam Collection)
    - Files with text and label columns

    Args:
        file_path: Path to the data file
        text_column: Name of the column containing text (auto-detected if None)
        label_column: Name of the column containing labels (auto-detected if None)
        delimiter: Delimiter character (auto-detected if None)

    Returns:
        Tuple of (texts, labels, label_names)
        - texts: List of text strings
        - labels: List of integer labels (0 or 1 for binary)
        - label_names: List of unique label names
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"Data file not found: {file_path}")

    # Auto-detect delimiter
    if delimiter is None:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            first_line = f.readline()
            if '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = ','

    # Try to read as CSV/TSV with pandas
    try:
        # Check if file has headers
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            first_line = f.readline().strip()

        # SMS Spam Collection format: label<tab>message (no header)
        if delimiter == '\t' and first_line.lower().startswith(('ham', 'spam')):
            # No header, read directly
            df = pd.read_csv(file_path, sep=delimiter, header=None, names=['label', 'text'],
                           encoding='utf-8', on_bad_lines='skip')
        else:
            # Has header
            df = pd.read_csv(file_path, sep=delimiter, encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

    # Auto-detect text column
    if text_column is None:
        text_candidates = ['text', 'message', 'content', 'body', 'sms', 'email', 'review', 'comment']
        for col in df.columns:
            if col.lower() in text_candidates:
                text_column = col
                break
        if text_column is None:
            # Use the column with longest average string length
            str_cols = df.select_dtypes(include=['object']).columns
            if len(str_cols) > 0:
                avg_lengths = {col: df[col].astype(str).str.len().mean() for col in str_cols}
                text_column = max(avg_lengths, key=avg_lengths.get)

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available columns: {list(df.columns)}")

    # Auto-detect label column
    if label_column is None:
        label_candidates = ['label', 'target', 'class', 'category', 'spam', 'sentiment']
        for col in df.columns:
            if col.lower() in label_candidates:
                label_column = col
                break
        if label_column is None:
            # Use column that's not the text column and has few unique values
            for col in df.columns:
                if col != text_column and df[col].nunique() <= 10:
                    label_column = col
                    break

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available columns: {list(df.columns)}")

    texts = df[text_column].astype(str).tolist()
    raw_labels = df[label_column].tolist()

    # Convert labels to integers
    unique_labels = sorted(list(set(raw_labels)))
    label_names = [str(l) for l in unique_labels]

    # Determine which label is positive (1) for binary classification
    positive_indicators = {'spam', '1', 'positive', 'yes', 'true', 'malicious', 'toxic', 'bad'}

    positive_idx = None
    for i, name in enumerate(label_names):
        if str(name).lower() in positive_indicators:
            positive_idx = i
            break

    # Default: second label alphabetically is positive
    if positive_idx is None and len(unique_labels) == 2:
        positive_idx = 1

    # Create integer labels
    label_map = {}
    for i, label in enumerate(unique_labels):
        if len(unique_labels) == 2:
            label_map[label] = 1 if i == positive_idx else 0
        else:
            label_map[label] = i

    labels = [label_map[l] for l in raw_labels]

    return texts, labels, label_names


def get_sample_texts(texts: List[str], labels: List[int],
                     n_per_class: int = 3) -> Dict[int, List[str]]:
    """
    Get sample texts from each class for display.

    Args:
        texts: List of all text strings
        labels: List of corresponding labels
        n_per_class: Number of samples per class

    Returns:
        Dict mapping label to list of sample texts
    """
    samples = {}
    for label in set(labels):
        class_texts = [t for t, l in zip(texts, labels) if l == label]
        samples[label] = class_texts[:n_per_class]
    return samples


# =============================================================================
# Dataset Hashing (for cache invalidation)
# =============================================================================

def get_text_key(text: str) -> str:
    """Create a unique key for a text (hash of first 200 chars + length)."""
    content = f"{text[:200]}_{len(text)}"
    return hashlib.md5(content.encode()).hexdigest()


def compute_dataset_hash(texts: List[str]) -> str:
    """
    Compute a hash of the dataset based on text content.
    Used to detect when dataset changes and cache should be invalidated.
    """
    content = "\n".join([f"{i}:{t[:100]}" for i, t in enumerate(texts[:100])])
    content += f"\n_total:{len(texts)}"
    return hashlib.md5(content.encode()).hexdigest()


def is_cache_valid(cache_dir: str, texts: List[str]) -> bool:
    """
    Check if the cache is still valid for this dataset.
    """
    cache_path = Path(cache_dir)
    hash_file = cache_path / DATASET_HASH_FILENAME

    if not hash_file.exists():
        return False

    stored_hash = hash_file.read_text().strip()
    current_hash = compute_dataset_hash(texts)

    return stored_hash == current_hash


def save_dataset_hash(cache_dir: str, texts: List[str]):
    """
    Save the current dataset hash for future cache validation.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    hash_file = cache_path / DATASET_HASH_FILENAME
    current_hash = compute_dataset_hash(texts)
    hash_file.write_text(current_hash)


# =============================================================================
# Tag Extraction (OpenAI API - Batched)
# =============================================================================

def extract_tags_batch(client, texts_batch: List[str],
                       dataset_description: str,
                       max_retries: int = 5) -> Tuple[Optional[List[List[str]]], Dict]:
    """
    Use OpenAI API to extract tags from a batch of text messages.

    Args:
        client: OpenAI client instance
        texts_batch: List of text messages to analyze (batch)
        dataset_description: Short description of the dataset for context
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (list of tag lists for each message, usage stats dict)
    """
    if not OPENAI_AVAILABLE:
        return None, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}

    # Format texts with indices
    formatted_texts = "\n".join([
        f"[{i}] {text[:500]}" for i, text in enumerate(texts_batch)  # Truncate long texts
    ])

    usage_stats = {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}

    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are analyzing text messages from this dataset: {dataset_description}

For EACH message in the batch (identified by [0], [1], etc.), extract relevant descriptive tags that could help classify the message.

Guidelines:
- Use lowercase tags with underscores (e.g., "has_url", "mentions_prize", "urgent_tone", "informal_language")
- Include both content-based tags (what the message contains) and style-based tags (how it's written)
- Be consistent with tag naming across messages
- Return ONLY tags, no explanations or other text
- Return tags for ALL messages in the exact order provided
- Focus on features that would help distinguish between classes"""
                    },
                    {
                        "role": "user",
                        "content": f"Extract tags for these {len(texts_batch)} messages:\n\n{formatted_texts}"
                    }
                ],
                response_format=BatchMessageTags,
            )

            # Extract usage statistics
            if response.usage:
                usage_stats['input_tokens'] = response.usage.prompt_tokens
                usage_stats['output_tokens'] = response.usage.completion_tokens

            parsed = response.choices[0].message.parsed

            # Extract just the tag lists
            tags_lists = [msg_tags.tags for msg_tags in parsed.results]

            # Verify we got the right number of results
            if len(tags_lists) != len(texts_batch):
                # Pad with empty lists if we got fewer
                while len(tags_lists) < len(texts_batch):
                    tags_lists.append([])

            return tags_lists, usage_stats

        except RateLimitError as e:
            if is_quota_error(e):
                raise QuotaExhaustedError(f"OpenAI quota exhausted: {e}")
            wait_time = (2 ** attempt) + (0.5 * attempt)
            time.sleep(wait_time)

            if attempt == max_retries - 1:
                return None, usage_stats

        except APIError as e:
            if is_quota_error(e):
                raise QuotaExhaustedError(f"OpenAI quota exhausted: {e}")
            wait_time = 2 * (attempt + 1)
            time.sleep(wait_time)

            if attempt == max_retries - 1:
                return None, usage_stats

        except Exception as e:
            return None, usage_stats

    return None, usage_stats


def is_valid_tag(tag: str) -> bool:
    """
    Check if a tag is valid (not message content or gibberish).
    Valid tags are short, lowercase, use underscores, no spaces in middle.
    """
    if not tag or not isinstance(tag, str):
        return False
    tag = tag.strip()

    # Too long - likely message content
    if len(tag) > 50:
        return False
    # Empty after strip
    if len(tag) == 0:
        return False
    # Contains multiple spaces (likely a sentence)
    if '  ' in tag or tag.count(' ') > 3:
        return False
    # Starts with capital and has spaces (likely a sentence)
    if tag[0].isupper() and ' ' in tag:
        return False
    # Check if normalized tag is a reserved column name
    normalized = tag.strip().lower().replace(' ', '_').replace('-', '_')
    normalized = ''.join(c for c in normalized if c.isalnum() or c == '_')
    if normalized in RESERVED_COLUMNS:
        return False

    # Check for conversational responses
    conversational_starts = [
        "I ", "I'", "I'm", "Please", "However", "If you", "Let me",
        "feel free", "you can", "including", "based on"
    ]
    for pattern in conversational_starts:
        if tag.lower().startswith(pattern.lower()):
            return False

    return True


def normalize_tag(tag: str) -> str:
    """Normalize a tag to lowercase with underscores."""
    tag = tag.strip().lower()
    tag = tag.replace(' ', '_').replace('-', '_')
    # Remove any non-alphanumeric chars except underscore
    tag = ''.join(c for c in tag if c.isalnum() or c == '_')
    return tag


def filter_tags(tags: List[str]) -> List[str]:
    """Filter and normalize tags."""
    valid_tags = [tag for tag in tags if is_valid_tag(tag)]
    normalized_tags = [normalize_tag(tag) for tag in valid_tags]
    return normalized_tags


def extract_and_cache_tags(cache_dir: str, texts: List[str],
                           dataset_description: str,
                           openai_api_key: Optional[str] = None,
                           force_refresh: bool = False,
                           batch_size: int = DEFAULT_BATCH_SIZE,
                           delay_between_batches: float = DEFAULT_DELAY_BETWEEN_BATCHES,
                           progress_callback=None) -> Dict[str, List[str]]:
    """
    Extract tags for all texts using batched API requests with caching.

    Args:
        cache_dir: Directory for cache storage
        texts: List of text messages
        dataset_description: Short description for the model
        openai_api_key: OpenAI API key (optional, uses env var if not provided)
        force_refresh: If True, re-extract all tags ignoring cache
        batch_size: Number of texts per API request
        delay_between_batches: Delay in seconds between batch requests
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary mapping text keys to their extracted tags
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / TAGS_CACHE_FILENAME

    # Load existing cache
    cache = {}
    if not force_refresh and cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)

    # Find texts that need processing
    texts_to_process = []
    for idx, text in enumerate(texts):
        text_key = get_text_key(text)
        if text_key not in cache:
            texts_to_process.append((idx, text, text_key))

    if not texts_to_process:
        # Filter existing cache
        filtered_cache = {k: filter_tags(v) for k, v in cache.items()}
        return filtered_cache

    # Check for OpenAI availability
    if not OPENAI_AVAILABLE:
        for idx, text, text_key in texts_to_process:
            cache[text_key] = []
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        return {k: filter_tags(v) for k, v in cache.items()}

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        for idx, text, text_key in texts_to_process:
            cache[text_key] = []
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        return {k: filter_tags(v) for k, v in cache.items()}

    client = OpenAI(api_key=api_key)

    # Calculate number of batches
    num_batches = (len(texts_to_process) + batch_size - 1) // batch_size

    # Process in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts_to_process))
        batch = texts_to_process[start_idx:end_idx]

        # Extract just the texts for this batch
        batch_texts = [text for _, text, _ in batch]

        # Call batched API
        try:
            tags_lists, usage_stats = extract_tags_batch(client, batch_texts, dataset_description)
        except QuotaExhaustedError:
            # Don't cache failed results, re-raise to notify user
            raise

        if tags_lists:
            # Store results in cache
            for (_, _, text_key), tags in zip(batch, tags_lists):
                cache[text_key] = tags
        else:
            # Store empty tags on failure
            for _, _, text_key in batch:
                cache[text_key] = []

        # Save cache after each batch
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)

        if progress_callback:
            progress_callback(end_idx, len(texts_to_process))

        # Delay between batches (except for the last one)
        if batch_idx < num_batches - 1 and delay_between_batches > 0:
            time.sleep(delay_between_batches)

    # Filter and return
    filtered_cache = {k: filter_tags(v) for k, v in cache.items()}
    return filtered_cache


# =============================================================================
# Tag Reduction and Merging
# =============================================================================

# Default number of tags to keep after reduction
DEFAULT_MAX_TAGS = 30


def merge_similar_tags(tags_cache: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict]:
    """
    Merge semantically similar tags using sentence transformers.

    Uses the shared tag_standardization module for semantic similarity-based
    clustering with sentence-transformers (MiniLM-L6-v2).

    Args:
        tags_cache: Dict mapping text keys to tag lists

    Returns:
        Tuple of (merged_cache, merge_map, stats)
        - merged_cache: Cache with similar tags merged to canonical forms
        - merge_map: Dict mapping original tags to their canonical form
        - stats: Standardization statistics
    """
    # Collect all unique tags and their frequencies
    tag_frequencies = {}
    for tags in tags_cache.values():
        for tag in tags:
            tag_frequencies[tag] = tag_frequencies.get(tag, 0) + 1

    all_tags = list(tag_frequencies.keys())

    if len(all_tags) < 2:
        return tags_cache, {t: t for t in all_tags}, {}

    # Use semantic standardization from shared module
    try:
        merge_map = standardize_tags_with_frequencies(
            all_tags,
            tag_frequencies,
            distance_threshold=TAG_SIMILARITY_THRESHOLD
        )
        merged_cache = apply_tag_mapping_to_cache(tags_cache, merge_map)
        stats = compute_standardization_stats(all_tags, merge_map)
    except ImportError:
        # Fallback: no merging if sentence-transformers not installed
        merge_map = {t: t for t in all_tags}
        merged_cache = tags_cache
        stats = {'skipped': True, 'reason': 'sentence-transformers not installed'}

    return merged_cache, merge_map, stats


def compute_tag_discriminative_power(tags_cache: Dict[str, List[str]],
                                      texts: List[str],
                                      labels: List[int]) -> Dict[str, float]:
    """
    Compute how discriminative each tag is for classification.

    Uses the absolute difference in tag frequency between classes
    (similar to correlation with label).

    Args:
        tags_cache: Dict mapping text keys to tag lists
        texts: List of text strings
        labels: List of labels (0 or 1)

    Returns:
        Dict mapping tag to its discriminative score (higher = more useful)
    """
    # Collect tag presence per class
    tag_counts_positive = {}
    tag_counts_negative = {}
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive

    for text, label in zip(texts, labels):
        text_key = get_text_key(text)
        tags = tags_cache.get(text_key, [])

        for tag in tags:
            if label == 1:
                tag_counts_positive[tag] = tag_counts_positive.get(tag, 0) + 1
            else:
                tag_counts_negative[tag] = tag_counts_negative.get(tag, 0) + 1

    # Compute discriminative power for each tag
    all_tags = set(tag_counts_positive.keys()) | set(tag_counts_negative.keys())
    discriminative_scores = {}

    for tag in all_tags:
        # Compute frequency in each class
        freq_positive = tag_counts_positive.get(tag, 0) / max(n_positive, 1)
        freq_negative = tag_counts_negative.get(tag, 0) / max(n_negative, 1)

        # Discriminative power = absolute difference in frequencies
        # Higher means the tag is more predictive of one class
        discriminative_scores[tag] = abs(freq_positive - freq_negative)

    return discriminative_scores


def select_top_tags(tags_cache: Dict[str, List[str]],
                    texts: List[str],
                    labels: List[int],
                    max_tags: int = DEFAULT_MAX_TAGS,
                    min_frequency: int = 2) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Select the top N most meaningful tags and filter the cache.

    Tags are selected based on:
    1. Minimum frequency (must appear at least min_frequency times)
    2. Discriminative power (ability to distinguish between classes)

    Args:
        tags_cache: Dict mapping text keys to tag lists
        texts: List of text strings
        labels: List of labels
        max_tags: Maximum number of tags to keep
        min_frequency: Minimum times a tag must appear to be considered

    Returns:
        Tuple of (selected_tags, filtered_cache)
        - selected_tags: List of selected tag names
        - filtered_cache: Cache with only selected tags
    """
    # Compute tag frequencies
    tag_frequencies = {}
    for tags in tags_cache.values():
        for tag in tags:
            tag_frequencies[tag] = tag_frequencies.get(tag, 0) + 1

    # Filter by minimum frequency
    frequent_tags = {tag for tag, freq in tag_frequencies.items() if freq >= min_frequency}

    # Compute discriminative power
    discriminative_scores = compute_tag_discriminative_power(tags_cache, texts, labels)

    # Filter scores to only include frequent tags
    filtered_scores = {
        tag: score for tag, score in discriminative_scores.items()
        if tag in frequent_tags
    }

    # Sort by discriminative power and select top N
    sorted_tags = sorted(filtered_scores.items(), key=lambda x: -x[1])
    selected_tags = [tag for tag, _ in sorted_tags[:max_tags]]

    # Create filtered cache with only selected tags
    selected_set = set(selected_tags)
    filtered_cache = {}
    for text_key, tags in tags_cache.items():
        filtered_cache[text_key] = [tag for tag in tags if tag in selected_set]

    return selected_tags, filtered_cache


def reduce_and_merge_tags(tags_cache: Dict[str, List[str]],
                          texts: List[str],
                          labels: List[int],
                          max_tags: int = DEFAULT_MAX_TAGS,
                          min_frequency: int = 2) -> Tuple[Dict[str, List[str]], List[str], Dict]:
    """
    Main function to reduce tags by merging similar ones and selecting top N.

    Pipeline:
    1. Merge semantically similar tags into canonical forms (using sentence transformers)
    2. Select top N most discriminative tags

    Args:
        tags_cache: Original tags cache from extraction
        texts: List of text strings
        labels: List of labels
        max_tags: Maximum number of tags to keep
        min_frequency: Minimum tag frequency to consider

    Returns:
        Tuple of (reduced_cache, selected_tags, reduction_stats)
    """
    # Step 1: Merge similar tags using semantic similarity
    merged_cache, merge_map, merge_stats = merge_similar_tags(tags_cache)

    # Count unique tags before and after merge
    original_tags = set()
    for tags in tags_cache.values():
        original_tags.update(tags)

    merged_tags = set()
    for tags in merged_cache.values():
        merged_tags.update(tags)

    # Step 2: Select top discriminative tags
    selected_tags, reduced_cache = select_top_tags(
        merged_cache, texts, labels,
        max_tags=max_tags,
        min_frequency=min_frequency
    )

    # Get merged clusters for reporting
    merged_clusters = get_merged_clusters(merge_map)

    # Compute stats
    reduction_stats = {
        'original_tag_count': len(original_tags),
        'after_merge_count': len(merged_tags),
        'final_tag_count': len(selected_tags),
        'merged_groups': len(merged_clusters),
        'merge_map_sample': dict(list(merge_map.items())[:10]),  # Sample of merges
        'merged_clusters_sample': dict(list(merged_clusters.items())[:5]),  # Sample of clusters
    }

    return reduced_cache, selected_tags, reduction_stats


# =============================================================================
# Convert to Tabular Dataset
# =============================================================================

def create_tabular_dataset(texts: List[str], labels: List[int],
                           tags_cache: Dict[str, List[str]],
                           max_tags: int = DEFAULT_MAX_TAGS) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Convert text dataset to tabular format using extracted tags as boolean features.

    Applies tag reduction:
    1. Merges similar tags into canonical forms
    2. Selects top N most discriminative tags

    Args:
        texts: List of text strings
        labels: List of labels
        tags_cache: Dict mapping text keys to tag lists
        max_tags: Maximum number of tags to keep (default: 30)

    Returns:
        Tuple of (DataFrame, selected_tags, reduction_stats)
    """
    # Apply tag reduction: merge similar tags and select top N
    reduced_cache, selected_tags, reduction_stats = reduce_and_merge_tags(
        tags_cache, texts, labels,
        max_tags=max_tags,
        min_frequency=2
    )

    # Create feature matrix using only selected tags
    data = []
    for text, label in zip(texts, labels):
        text_key = get_text_key(text)
        text_tags = set(reduced_cache.get(text_key, []))

        # Create boolean features for each selected tag
        row = {tag: (tag in text_tags) for tag in selected_tags}
        row['label'] = label
        row['text'] = text[:200]  # Store truncated text for reference

        data.append(row)

    df = pd.DataFrame(data)

    # Convert boolean columns to int
    for col in selected_tags:
        df[col] = df[col].astype(int)

    return df, selected_tags, reduction_stats


# =============================================================================
# Main API Function
# =============================================================================

def load_text_dataset_as_tabular(file_path: str,
                                  dataset_description: str,
                                  text_column: Optional[str] = None,
                                  label_column: Optional[str] = None,
                                  openai_api_key: Optional[str] = None,
                                  force_refresh: bool = False,
                                  batch_size: int = DEFAULT_BATCH_SIZE,
                                  max_tags: int = DEFAULT_MAX_TAGS,
                                  progress_callback=None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a text dataset and convert it to tabular format.

    This is the main entry point for text dataset loading. It:
    1. Reads the text file
    2. Extracts tags using OpenAI API (with caching and batching)
    3. Merges similar tags and selects top N most discriminative
    4. Creates a tabular DataFrame with boolean tag columns

    Args:
        file_path: Path to the text data file
        dataset_description: Short description of the dataset for the model
        text_column: Name of column containing text (auto-detected if None)
        label_column: Name of column containing labels (auto-detected if None)
        openai_api_key: OpenAI API key (optional, uses env var if not provided)
        force_refresh: If True, ignore cache and re-extract everything
        batch_size: Number of texts per API batch request
        max_tags: Maximum number of tags to keep after reduction (default: 30)
        progress_callback: Callback for extraction progress

    Returns:
        Tuple of (DataFrame, metadata_dict)
        - DataFrame: Tabular dataset with boolean tag columns and label
        - metadata_dict: Contains label_names, all_tags, reduction_stats, etc.
    """
    # Load texts and labels
    texts, labels, label_names = load_text_dataset(
        file_path,
        text_column=text_column,
        label_column=label_column
    )

    # Create cache directory based on file name
    file_path = Path(file_path)
    cache_dir = file_path.parent / f".{file_path.stem}_cache"

    # Check cache validity
    cache_valid = is_cache_valid(str(cache_dir), texts) and not force_refresh

    # Extract tags
    tags_cache = extract_and_cache_tags(
        str(cache_dir), texts,
        dataset_description=dataset_description,
        openai_api_key=openai_api_key,
        force_refresh=not cache_valid,
        batch_size=batch_size,
        progress_callback=progress_callback
    )

    # Create tabular dataset (with tag reduction)
    df, selected_tags, reduction_stats = create_tabular_dataset(
        texts, labels, tags_cache, max_tags=max_tags
    )

    # Save dataset hash for future cache validation
    save_dataset_hash(str(cache_dir), texts)

    metadata = {
        'label_names': label_names,
        'all_tags': selected_tags,
        'n_texts': len(texts),
        'n_positive': sum(labels),
        'n_negative': len(labels) - sum(labels),
        'text_column': 'text',
        'label_column': 'label',
        'reduction_stats': reduction_stats,
    }

    return df, metadata
