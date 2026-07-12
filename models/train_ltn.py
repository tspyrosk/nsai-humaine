import ltn
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score, precision_score, recall_score, f1_score
import random
import argparse
import time
import json
import warnings
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from sklearn.model_selection import train_test_split
import sys
import os
from paths import *
from dataset import minio_utils 
from utils import normalize_value, random_oversample

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_path", type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--use_rules", default=0, type=int)
parser.add_argument("--rules_file_path", default="", type=str)

args = parser.parse_args()

input_path = args.input_path
batch_size = args.batch_size
epochs = args.epochs
RANDOM_SEED = args.seed
USE_RULES = args.use_rules
rules_file_path = args.rules_file_path

if USE_RULES:
    model_name = 'ltn'
else:
    model_name = 'mlp'

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)
# Workaround for tf_keras bug: backend._create_seed calls
# _SEED_GENERATOR.generator.randint(1, 1e9) with a Python float, which
# random.Random.randint rejects ("'float' object cannot be interpreted as an integer").
try:
    from tf_keras.src import backend as _tfk_backend
    _gen = getattr(_tfk_backend._SEED_GENERATOR, "generator", None)
    if _gen is not None:
        _orig_randint = _gen.randint
        _gen.randint = lambda a, b, _f=_orig_randint: _f(int(a), int(b))
except Exception:
    pass
tf.config.experimental.enable_op_determinism()
METADATA = {}

def extract_layer_details(model):
    layer_details = []
    for i, layer in enumerate(model.layers):
        config = layer.get_config()
        layer_details.append({
            "layer_index": i,
            "layer_type": layer.__class__.__name__,
            "config": config,
            "next_layers": []
        })
    return layer_details

X_train = np.load(f'{input_path}/train/X_{RANDOM_SEED}.npy')
y_train = np.load(f'{input_path}/train/y_{RANDOM_SEED}.npy')
X_test = np.load(f'{input_path}/test/X_{RANDOM_SEED}.npy')
y_test = np.load(f'{input_path}/test/y_{RANDOM_SEED}.npy')

train_idx = np.load(f'{input_path}/train/X_{RANDOM_SEED}.npy')
test_idx = np.load(f'{input_path}/test/X_{RANDOM_SEED}.npy')
METADATA["Training Indices"] = train_idx.tolist()
METADATA["Validation Indices"] = test_idx.tolist()
METADATA["Transformation Steps"] = ["Random Oversampling", "Normalize"]

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((X_test,y_test)).batch(batch_size)

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=2),semantics="exists")


formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))


with open(f"{OUTPUT_DIR}/predicates.txt", 'r') as file:
    preds_code = file.read()
exec(preds_code, globals())

global f

f = ltn.Predicate.MLP([(X_train.shape[1],)],hidden_layer_sizes=(8,8))

def get_axioms(X_train, y_train, use_rules):
    x = ltn.Variable("x", X_train)
    y = ltn.Variable("y", y_train)
    x_A = ltn.Variable("x_A", tf.boolean_mask(X_train, y_train))
    x_not_A = ltn.Variable("x_not_A", tf.boolean_mask(X_train, tf.logical_not(tf.cast(y_train, tf.bool))))
    base_axioms = [
        Forall(x_A, target(x_A, f(x_A))),
        Forall(x_not_A, no_target(x_not_A, f(x_not_A)))
    ]

    axioms = base_axioms
    if use_rules:
        with open(rules_file_path, 'r') as file:
            rules_from_file = file.read()
        exec(rules_from_file, globals())
        axioms = base_axioms + parsed_rules(x, y)
    return axioms


def train(X_train_full, y_train_full, epochs, use_rules, log=False):
    METADATA["Optimizer"] = {
        "type": "Adam",
        "config": {"lr": 0.001},
        "loss_metric": "sat"
    }
    METADATA["Rules File"] = "local" if use_rules else None
    METADATA["Model Type"] = "LTN"
    METADATA["Model Version"] = "v0.1"

    METADATA["Hyperparameters"] = {
        "learning_rate": 0.001,
        "batch_size": 1,
        "epochs": epochs
    },
    METADATA["Loss Metric"] = "sat"
    METADATA["Knowledge Representation"] = "LTN"
    METADATA["Epoch Statistics"] = {}
    METADATA["Satisfiability Scores"] = {}

    indices = np.arange(len(X_train_full))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_train, y_train = random_oversample(X_train, y_train, random_state=RANDOM_SEED)

    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    METADATA["Training Indices"] = train_idx.tolist()
    METADATA["Validation Indices"] = val_idx.tolist()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    losses = []
    val_losses = []
    val_loss_min = 10e6
    last_saved_epoch = 0

    start_time = time.time()
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            axioms = get_axioms(X_train, y_train, use_rules)
            sat_level = formula_aggregator(axioms).tensor
            loss = 1. - sat_level
        gradients = tape.gradient(loss, f.trainable_variables)
        optimizer.apply_gradients(zip(gradients, f.trainable_variables))
        losses.append(loss.numpy())

        val_axioms = get_axioms(X_val, y_val, use_rules)
        val_sat = formula_aggregator(val_axioms).tensor
        val_loss = 1. - val_sat
        val_losses.append(val_loss.numpy())

        if val_loss < val_loss_min:
            if log:
                print(f'Epoch: {epoch}: Val loss reduced to {val_loss}. Saving model.')
            f.model.save(f'{OUTPUT_DIR}/{model_name}.h5')
            val_loss_min = val_loss
            last_saved_epoch = epoch

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}')
            print(f'Train Sat: {sat_level.numpy():.4f}')
            print(f'Val Sat: {val_sat.numpy():.4f}')
            print(f'Last Saved Epoch: {last_saved_epoch}')
            METADATA["Epoch Statistics"][f"epoch_{epoch}"] = {"train_loss": str(loss.numpy()),
                                                              "val_loss": str(val_loss.numpy())}
            METADATA["Satisfiability Scores"][f"epoch_{epoch}"] = {"train_sat": str(sat_level.numpy()),
                                                                   "val_sat": str(val_sat.numpy())}


    end_time = time.time()

    total_time = end_time - start_time
    METADATA["Training Time"] = total_time
    METADATA["Training Time per Epoch"] = total_time / epochs
    METADATA["Layer Details"] = extract_layer_details(f.model)


train(X_train, y_train, epochs=epochs, use_rules=USE_RULES)
#871

saved_model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{model_name}.h5')

y_pred = saved_model(X_test).numpy()
preds = (y_pred > 0.5).astype(int).flatten()

report = classification_report(y_test, preds)
bal_acc = balanced_accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

val_failures = np.where(preds != y_test)[0]
METADATA["Post-training Evaluation Metrics"] = {
    "balanced accuracy": bal_acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

METADATA["Training Failure Cases"] = []
METADATA["Validation Failure Cases"] = val_failures.tolist()
print(report)
print("Balanced ACC: ", bal_acc)

if USE_RULES:
    timestr = time.strftime("%Y%m%d_%H%M%S")

    bucket_name = "smart-healthcare-diabetes-models"
    object_name = f"metadata/{timestr}_metadata.json"
    file_path = f"{OUTPUT_DIR}/metadata.json"

    with open(file_path, 'a') as f:
        f.write(json.dumps(METADATA) + '\n')

    token = minio_utils.minio_auth(os.getenv("MINIO_USER"), os.getenv("MINIO_PASS"))
    minio_utils.minio_upload(token, bucket_name, object_name, file_path)
    minio_utils.minio_upload(token, bucket_name, f"test-models/{timestr}_diabetes_ltn.h5", f"{OUTPUT_DIR}/ltn.h5")
    minio_utils.minio_upload(token, bucket_name, f"test-models/{timestr}_scaler.pkl", f"{OUTPUT_DIR}/scaler.pkl")