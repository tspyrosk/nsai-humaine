"""Import an NSAI rule set from a fitted scikit-learn decision tree.

Unlike the text-based importers there is no grammar: every root-to-leaf path of
a ``DecisionTreeClassifier`` is a conjunction of threshold tests ending in a
class, which is exactly the canonical rule model. Each split ``X[col] <= t``
becomes a ``Less`` predicate on the left branch and a ``Greater`` predicate on
the right branch; a predicate is emitted once per distinct (feature, threshold,
direction) and reused across paths.

A rule is produced for every leaf whose majority class is the positive class.
The accepted input is a fitted estimator, or a ``.pkl`` / ``.joblib`` file (or
bytes) containing one — including a sklearn ``Pipeline`` whose final step is the
tree.

Note: sklearn splits on ``<=``; the model's ``Less`` predicate uses ``<``. For
continuous features the distinction is immaterial.
"""

import io
import pickle
import re

import joblib

from rules_parsing import canonical


def _sanitize(value):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_").lower()


def _unwrap(model):
    """Return the underlying tree estimator from a raw model / pipeline / dict."""
    if hasattr(model, "tree_"):
        return model
    # sklearn Pipeline: take the final fitted step.
    if hasattr(model, "steps"):
        final = model.steps[-1][1]
        if hasattr(final, "tree_"):
            return final
    if isinstance(model, dict):
        for v in model.values():
            if hasattr(v, "tree_"):
                return v
    raise ValueError("Object does not contain a fitted decision-tree classifier")


def import_tree(clf, feature_names=None, positive_class=None, target_name="target"):
    clf = _unwrap(clf)
    tree = clf.tree_
    classes = list(clf.classes_)
    if positive_class is None:
        positive_class = classes[-1]
    if positive_class not in classes:
        raise ValueError(f"positive_class {positive_class!r} not in {classes}")
    pos_idx = classes.index(positive_class)

    predicates = {}  # name -> predicate dict (deduplicated, insertion-ordered)
    rules = []

    def feature_label(col):
        if feature_names is not None and col < len(feature_names):
            return _sanitize(feature_names[col])
        return f"feature_{col}"

    def add_predicate(col, threshold, direction):
        thr_str = _sanitize(f"{threshold:.4g}")
        name = f"{feature_label(col)}_{direction}_{thr_str}"
        if name not in predicates:
            predicates[name] = {
                "name": name,
                "column_index": int(col),
                "threshold": float(threshold),
                "comparison": "Less" if direction == "le" else "Greater",
                "is_boolean": False,
            }
        return name

    def recurse(node, conds):
        if tree.children_left[node] == -1:  # leaf
            leaf_class = tree.value[node][0].argmax()
            if leaf_class == pos_idx and conds:
                body = canonical.fold_and([canonical.compound(n) for n in conds])
                rules.append({
                    "if_part": canonical.body_to_rule_part(body),
                    "then_part": {"name": target_name},
                })
            return
        col = tree.feature[node]
        thr = tree.threshold[node]
        recurse(tree.children_left[node], conds + [add_predicate(col, thr, "le")])
        recurse(tree.children_right[node], conds + [add_predicate(col, thr, "gt")])

    recurse(0, [])

    return {
        "predicates": list(predicates.values()),
        "composite_predicates": [],
        "rules": rules,
    }


def import_tree_obj(data, **kwargs):
    """Load a model from a file-like / bytes object and import it."""
    if isinstance(data, (bytes, bytearray)):
        data = io.BytesIO(data)
    try:
        model = joblib.load(data)
    except Exception:
        data.seek(0)
        model = pickle.load(data)
    return import_tree(model, **kwargs)


def import_tree_file(path, **kwargs):
    with open(path, "rb") as f:
        return import_tree_obj(f, **kwargs)
