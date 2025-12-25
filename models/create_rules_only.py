import numpy as np
import random

from utils import normalize_value
import os
import re
from paths import *

def get_highest_seed_file(directory=TRAIN_DATA_DIR, prefix='X_', extension='.npy'):
    files = os.listdir(directory)
    pattern = re.compile(f"{prefix}(\\d+){re.escape(extension)}")

    seeds = []
    for file in files:
        match = pattern.match(file)
        if match:
            seeds.append(int(match.group(1)))

    if not seeds:
        raise FileNotFoundError("No matching seed files found.")

    max_seed = max(seeds)
    filename = f"{prefix}{max_seed}{extension}"
    return np.load(os.path.join(directory, filename)), max_seed


X_train, RANDOM_SEED = get_highest_seed_file()

with open(f"{OUTPUT_DIR}/lambdas.txt", 'r') as file:
    lambdas_code = file.read()
exec(lambdas_code, locals())

def get_concepts():
    return {
        var_name.replace("_", " ").capitalize(): var
        for var_name, var in locals().items()
        if any(var is lv for lv in lambda_vars)
    }

def satisfied_concepts(x):
    return [name for name, condition in get_concepts().items() if condition(x)]

with open(f"{OUTPUT_DIR}/rules_only_rules.txt", 'r') as file:
    rules_from_file = file.read().replace("```", "").replace("`", "")

exec(rules_from_file, globals())
rules = parsed_rules_python

def predict(X, Y):
    preds = []
    for x, y in zip(X, Y):
        evaluated_rules = rules(x, y)
        catch_all_rule = (True,random.choice([0, 1]))
        evaluated_rules.append(catch_all_rule)
        for rule, result in evaluated_rules:
            if(rule):
                preds.append(result)
                break

    return np.array(preds)

def get_satisfied_rule_indexes(x, y):
    indexes = []

    evaluated_rules = rules(x, y)
    for idx, (rule, _) in enumerate(evaluated_rules):
        if(rule):
            indexes.append(idx)

    return np.array(indexes)