import numpy as np
import re
import os
import shutil
from paths import *

def normalize_value(features, feature_index, value):
    feature_column = features[:, feature_index]

    mean = np.mean(feature_column)
    std = np.std(feature_column)
    if std == 0:
        raise ValueError("Standard deviation is zero. Cannot normalize.")
    return (value - mean) / std

def random_oversample(X, y, random_state=None):
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)

    class_counts = {cls: np.sum(y == cls) for cls in classes}
    max_count = max(class_counts.values())

    X_resampled = list(X)
    y_resampled = list(y)

    for cls in classes:
        count = class_counts[cls]
        n_samples_to_add = max_count - count

        if n_samples_to_add > 0:
            indices = np.where(y == cls)[0]
            sampled_indices = rng.choice(indices, size=n_samples_to_add, replace=True)
            X_resampled.extend(X[i] for i in sampled_indices)
            y_resampled.extend(y[i] for i in sampled_indices)

    return np.array(X_resampled), np.array(y_resampled)

def to_snake_case(name):
    return re.sub(r"\s+", "_", name.lower())

def collect_predicate_names(single, composite, target):
    names = [to_snake_case(s["name"]) for s in single]
    composite_names = [to_snake_case(c["name"]) for c in composite]
    names.extend(composite_names)
    names.append(to_snake_case(target))
    return names

def remove_outputs():
    directory = OUTPUT_DIR

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove subdirectory
            print(f"Removed {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
    
def remove_files(files_to_remove):
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed {f}")

def remove_rules_and_predicates():
    dir = OUTPUT_DIR
    file_names = [
        "lambdas.txt",
        "predicates.txt",
        "rules_raw.txt",
        "ltn_rules.txt",
        "rules_only_rules.txt"
    ]
    files_to_remove = [f"{dir}/{name}" for name in file_names]
    remove_files(files_to_remove)

def remove_rules():
    dir = OUTPUT_DIR
    file_names = [
        "rules_raw.txt",
        "ltn_rules.txt",
        "rules_only_rules.txt"
    ]
    files_to_remove = [f"{dir}/{name}" for name in file_names]
    remove_files(files_to_remove)
    