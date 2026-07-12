import os

#BASE PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

#INPUT PATHS
INPUT_CSV = os.path.join(INPUT_DIR, "input.csv")

#OUTPUT PATHS
FEATURE_NAMES_PATH = os.path.join(OUTPUT_DIR, "feature_names.txt")
TAG_VOCABULARY_PATH = os.path.join(OUTPUT_DIR, "tag_vocabulary.json")
TRAIN_DATA_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DATA_DIR = os.path.join(OUTPUT_DIR, "test")
LTN_RULES_PATH = os.path.join(OUTPUT_DIR, "ltn_rules.txt")

#NOTEBOOK PATHS
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
JUPYTER_URL = os.getenv("JUPYTER_URL", "/jupyter")

#SCRIPT PATHS
SPLIT_DATASET_SCRIPT = os.path.join(BASE_DIR, "dataset", "split_dataset.py")