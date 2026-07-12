"""Spec for the additional rule-format importers.

Each text-format importer reads a fixture written in that format and returns the
same canonical dict the Prolog importer produces (see
``tests/test_prolog_importer.py``). Datalog, CLIPS and Drools express the full
example (matching ``example_rules.json``); SWRL is conjunction-only and matches
``example_rules_swrl.json``. The decision-tree importer has no fixture — a tiny
tree is fitted in-test — and is checked structurally.
"""

import json
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


@pytest.mark.parametrize("fixture, importer_path, expected", [
    ("example_rules.dl", "rules_parsing.datalog_importer:import_datalog_file", "example_rules.json"),
    ("example_rules.clp", "rules_parsing.clips_importer:import_clips_file", "example_rules.json"),
    ("example_rules.drl", "rules_parsing.drools_importer:import_drools_file", "example_rules.json"),
    ("example_rules.swrl", "rules_parsing.swrl_importer:import_swrl_file", "example_rules_swrl.json"),
])
def test_text_format_converts_to_expected_json(fixture, importer_path, expected):
    module_name, func_name = importer_path.split(":")
    module = __import__(module_name, fromlist=[func_name])
    import_file = getattr(module, func_name)

    result = import_file(str(FIXTURES / fixture))

    assert result == _load(expected)


def test_decision_tree_converts_to_canonical_rules():
    from sklearn.datasets import load_breast_cancer
    from sklearn.tree import DecisionTreeClassifier
    from rules_parsing.tree_importer import import_tree

    X, y = load_breast_cancer(return_X_y=True)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    result = import_tree(clf, positive_class=1, target_name="malignant")

    assert result["composite_predicates"] == []
    assert result["predicates"], "expected at least one threshold predicate"
    assert result["rules"], "expected at least one rule for the positive class"

    pred_names = {p["name"] for p in result["predicates"]}
    for p in result["predicates"]:
        assert p["comparison"] in ("Less", "Greater")
        assert isinstance(p["column_index"], int)
        assert isinstance(p["threshold"], float)

    for rule in result["rules"]:
        assert rule["then_part"] == {"name": "malignant"}
        _assert_leaves_are_known_predicates(rule["if_part"], pred_names)


def _assert_leaves_are_known_predicates(node, pred_names):
    op = node.get("operator")
    if op in ("AND", "OR"):
        _assert_leaves_are_known_predicates(node["arg1"], pred_names)
        _assert_leaves_are_known_predicates(node["arg2"], pred_names)
    else:
        assert node["name"] in pred_names


@pytest.mark.parametrize("importer_path, fixture", [
    ("rules_parsing.datalog_importer:import_datalog_file", "example_rules.dl"),
    ("rules_parsing.clips_importer:import_clips_file", "example_rules.clp"),
    ("rules_parsing.drools_importer:import_drools_file", "example_rules.drl"),
    ("rules_parsing.swrl_importer:import_swrl_file", "example_rules.swrl"),
])
def test_imported_rules_render_without_error(importer_path, fixture):
    """The shared renderers must accept rules from every importer."""
    from rules_parsing import canonical

    module_name, func_name = importer_path.split(":")
    module = __import__(module_name, fromlist=[func_name])
    parsed = getattr(module, func_name)(str(FIXTURES / fixture))

    for rule in parsed["rules"]:
        assert canonical.rule_to_text(rule).startswith("IF ")
        assert canonical.rule_to_ltn(rule).startswith("Forall(x,")
        assert canonical.rule_to_python_lambda(rule).startswith("(")
