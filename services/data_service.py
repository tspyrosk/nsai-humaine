"""
Data Service - Handles all data loading and preprocessing operations.
"""
import os
import numpy as np
import pandas as pd
import dataset.minio_utils as minio_utils
from paths import *


def load_data_from_csv(csv_path: str, target_column: str, columns_to_drop: list,
                        seed: int, sample_ratio: float) -> tuple:
    """
    Load data from CSV file and split into train/test sets.

    Args:
        csv_path: Path to the CSV file
        target_column: Name of the target column
        columns_to_drop: List of column names to drop
        seed: Random seed for reproducibility
        sample_ratio: Ratio of data to use (0.1 to 1.0)

    Returns:
        Tuple of (X_train, X_test, y_test, processed_df, test_indices)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    drop = ",".join(columns_to_drop)
    os.system(f"python {SPLIT_DATASET_SCRIPT} --input_file_path={csv_path} --output_path={OUTPUT_DIR} --seed={seed} --sample_ratio={sample_ratio} --target_column=\"{target_column}\" --drop=\"{drop}\"")

    X_train = np.load(f'{TRAIN_DATA_DIR}/X_{seed}.npy')
    X_test = np.load(f'{TEST_DATA_DIR}/X_{seed}.npy')
    y_test = np.load(f'{TEST_DATA_DIR}/y_{seed}.npy')
    test_indices = np.load(f'{TEST_DATA_DIR}/idx_{seed}.npy')

    # Load feature names and create processed dataframe
    with open(FEATURE_NAMES_PATH, 'r') as file:
        new_cols = [line.strip() for line in file if line.strip()]
    processed_df = pd.DataFrame(
        np.concatenate((X_train, X_test), axis=0),
        columns=new_cols
    )

    return X_train, X_test, y_test, processed_df, test_indices


def load_data_from_minio(minio_token: str, minio_path: str, target_column: str,
                          columns_to_drop: list, seed: int, sample_ratio: float) -> tuple:
    """
    Load data from MinIO and split into train/test sets.

    Args:
        minio_token: MinIO authentication token
        minio_path: MinIO path in format "bucket/path/to/file"
        target_column: Name of the target column
        columns_to_drop: List of column names to drop
        seed: Random seed for reproducibility
        sample_ratio: Ratio of data to use (0.1 to 1.0)

    Returns:
        Tuple of (X_train, X_test, y_test, processed_df, test_indices)

    Raises:
        Exception: If MinIO download fails
    """
    bucket = minio_path.split("/")[0]
    path = "/".join(minio_path.split("/")[1:])

    os.makedirs(INPUT_DIR, exist_ok=True)
    minio_utils.minio_download(minio_token, bucket, path, INPUT_CSV)

    return load_data_from_csv(INPUT_CSV, target_column, columns_to_drop, seed, sample_ratio)


def get_feature_names() -> list:
    """
    Get the list of feature names from the saved feature names file.

    Returns:
        List of feature names, or empty list if file doesn't exist
    """
    path = FEATURE_NAMES_PATH
    if os.path.exists(path):
        with open(path, 'r') as f:
            loaded_features = [line.strip() for line in f]
            return loaded_features
    return []


def save_uploaded_csv(uploaded_file_content: bytes, csv_path: str) -> pd.DataFrame:
    """
    Save uploaded CSV file and return as DataFrame.

    Args:
        uploaded_file_content: File content as bytes
        csv_path: Path where to save the CSV

    Returns:
        DataFrame containing the CSV data
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file_content)
    return pd.read_csv(csv_path)
