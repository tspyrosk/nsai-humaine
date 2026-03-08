import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import argparse
import joblib

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_file_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--sample_ratio", default=1.0, type=float)
parser.add_argument("--target_column", type=str)
parser.add_argument('--drop', type=str, help='Comma-separated list of columns to drop')

args = parser.parse_args()
cols_to_drop = args.drop.split(',') if args.drop else []

RANDOM_SEED = args.seed
SAMPLE_RATIO = args.sample_ratio
INPUT_FILE_PATH = args.input_file_path
OUTPUT_PATH = args.output_path
label_column = args.target_column

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def to_float_or_default(x, default=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

df = pd.read_csv(INPUT_FILE_PATH).sample(frac=SAMPLE_RATIO, random_state=RANDOM_SEED).reset_index(drop=True)

if len(cols_to_drop) > 0:
    df = df.drop(columns=cols_to_drop)
X = df.drop(columns=[label_column]).map(to_float_or_default)
y = df[label_column]


features_file_path = f'{OUTPUT_PATH}/feature_names.txt'
with open(features_file_path, 'w') as f:
    for feature in X.columns:
        f.write(feature + '\n')

X = X.to_numpy()
y = y.to_numpy()

indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, f'{OUTPUT_PATH}/scaler.pkl')

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("Training samples: ", y_train.shape[0])
print("Test samples: ", y_test.shape[0])

np.save(f'{OUTPUT_PATH}/train/X_{RANDOM_SEED}.npy', X_train)
np.save(f'{OUTPUT_PATH}/train/y_{RANDOM_SEED}.npy', y_train)
np.save(f'{OUTPUT_PATH}/train/idx_{RANDOM_SEED}.npy', train_idx)

np.save(f'{OUTPUT_PATH}/test/X_{RANDOM_SEED}.npy', X_test)
np.save(f'{OUTPUT_PATH}/test/y_{RANDOM_SEED}.npy', y_test)
np.save(f'{OUTPUT_PATH}/test/idx_{RANDOM_SEED}.npy', test_idx)
