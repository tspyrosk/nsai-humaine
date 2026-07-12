"""
Predicate Service - Handles predicate and rule generation and management.
"""
import os
import json
from predicates_definition.predicate_gen_utils import generate_python_code
import predicates_definition.lambda_gen_utils as lambda_gen_utils
from rules_parsing import (
    canonical,
    text2rules_core,
    prolog_importer,
    datalog_importer,
    swrl_importer,
    clips_importer,
    drools_importer,
    tree_importer,
)
from utils import collect_predicate_names
from paths import *


def generate_and_save_predicates(target_column: str, predicates: list,
                                  composite_predicates: list) -> tuple:
    """
    Generate Python code for predicates and save to files.

    Args:
        target_column: Name of the target column
        predicates: List of simple predicate definitions
        composite_predicates: List of composite predicate definitions

    Returns:
        Tuple of (predicate_code, lambda_code, predicate_names)
    """
    # Generate code
    predicate_code = generate_python_code(target_column, predicates, composite_predicates)
    lambda_code = lambda_gen_utils.generate_python_code(target_column, predicates, composite_predicates)

    # Save to files
    with open(f"{OUTPUT_DIR}/predicates.txt", "w") as file:
        file.write(predicate_code)

    with open(f"{OUTPUT_DIR}/lambdas.txt", "w") as file:
        file.write(lambda_code)

    # Generate and save predicate names
    predicate_names = collect_predicate_names(predicates, composite_predicates, target_column)
    with open(f"{OUTPUT_DIR}/predicate_names.txt", "w") as file:
        file.write("\n".join(predicate_names))

    return predicate_code, lambda_code, predicate_names


def check_predicate_name_exists(name: str, predicates: list,
                                 composite_predicates: list) -> bool:
    """
    Check if a predicate name already exists.

    Args:
        name: Name to check
        predicates: List of simple predicates
        composite_predicates: List of composite predicates

    Returns:
        True if name exists, False otherwise
    """
    pred_names = [p['name'] for p in predicates]
    comp_pred_names = [c['name'] for c in composite_predicates]
    return name in pred_names or name in comp_pred_names


RULES_JSON_PATH = f"{OUTPUT_DIR}/rules.json"


def save_rule_set(predicates: list, composite_predicates: list, rules: list) -> None:
    """Persist the canonical rule set to ``output/rules.json``.

    This is the single source of truth the publish/export flow reads, written
    whenever rules are created (manual authoring or file import).
    """
    with open(RULES_JSON_PATH, "w") as f:
        json.dump({
            "predicates": predicates,
            "composite_predicates": composite_predicates,
            "rules": rules,
        }, f, indent=2)


def load_rule_set() -> dict:
    """Return the canonical rule set from ``output/rules.json``, or ``None``."""
    if not os.path.exists(RULES_JSON_PATH):
        return None
    with open(RULES_JSON_PATH) as f:
        return json.load(f)


def parse_rule_text(nl_text: str, predicate_names: list) -> dict:
    """Parse one natural-language rule into the canonical structured rule dict.

    Called at rule-authoring time so the UI can immediately show the user the
    encoded form of what the LLM understood. Raises on parse/LLM failure.
    """
    return text2rules_core.parse_rule(nl_text, predicate_names)


def save_rules_artifacts(predicates: list, composite_predicates: list,
                         rules: list) -> None:
    """Write every rule artifact the training/publish steps consume.

    ``rules`` is a list of ``{"text": <display string>, "structured": <rule dict>}``
    entries. Produces ``rules_raw.txt``, ``ltn_rules.txt``, ``rules_only_rules.txt``
    and the canonical ``rules.json`` — the same file interface the batch LLM
    flow and the format importers have always written.
    """
    structured = [r["structured"] for r in rules]
    ltn_rule_strs = [canonical.rule_to_ltn(r) for r in structured]
    py_rule_strs = [canonical.rule_to_python_lambda(r) for r in structured]

    ltn_code = f"parsed_rules = lambda x, y: [{', '.join(ltn_rule_strs)}]"
    rules_code = f"parsed_rules_python = lambda x, y: [{', '.join(py_rule_strs)}]"

    with open(f"{OUTPUT_DIR}/ltn_rules.txt", "w") as f:
        f.write(ltn_code)
    with open(f"{OUTPUT_DIR}/rules_only_rules.txt", "w") as f:
        f.write(rules_code)
    with open(f"{OUTPUT_DIR}/rules_raw.txt", "w") as f:
        f.write("\n".join(r["text"] for r in rules))

    save_rule_set(predicates, composite_predicates, structured)


def _structured_rule_names(rule: dict) -> set:
    """Collect every predicate name referenced anywhere in a structured rule."""
    names = set()

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get("name"):
            names.add(node["name"])
        for key in ("if_part", "then_part", "arg1", "arg2"):
            if node.get(key):
                walk(node[key])

    walk(rule)
    return names


def find_references(name: str, composite_predicates: list, rules: list) -> list:
    """Return human-readable descriptions of everything referencing a predicate.

    Used to block removal of a predicate that a composite expression or a
    defined rule still depends on. Composite expressions store the original
    predicate names; structured rules store the snake-cased names.
    """
    from utils import to_snake_case

    referrers = []
    for comp in composite_predicates:
        if comp["name"] == name:
            continue
        leaves = set()

        def collect(node):
            if node["args"]:
                for arg in node["args"]:
                    collect(arg)
            else:
                leaves.add(node["name"])

        collect(canonical.parse_composite_expression(comp["expression"]))
        if name in leaves:
            referrers.append(f"composite predicate '{comp['name']}'")

    snake_name = to_snake_case(name)
    for rule in rules:
        if snake_name in _structured_rule_names(rule["structured"]):
            referrers.append(f"rule '{rule['text']}'")
    return referrers


def _finalize_import(parsed: dict, target_column: str) -> dict:
    """Turn a parsed canonical rule dict into the on-disk artifacts + UI payload.

    Shared by every format importer. Populates the same output artifacts
    (predicates.txt, lambdas.txt, predicate_names.txt, ltn_rules.txt,
    rules_only_rules.txt) that the interactive UI flow produces, so training can
    start immediately.

    Returns the parsed dict augmented with ``rule_texts`` (human-readable
    ``IF ... THEN ...`` strings for display) and ``ltn_code`` / ``rules_code``
    (generated source the UI surfaces back to the user).
    """
    from utils import to_snake_case

    # Rule heads in the imported file (e.g. `target(X) :- ...`) need to resolve
    # to the lambda that lambda_gen_utils.py writes under the snake-cased target
    # column name (e.g. `malignant`). Remap every rule's then_part name here so
    # downstream rendering hits the right symbol.
    target_lambda_name = to_snake_case(target_column)
    for rule in parsed["rules"]:
        rule["then_part"]["name"] = target_lambda_name

    # Reuse the existing predicate-file generator so the imported set goes
    # through the same code path as UI-defined predicates.
    predicate_code, lambda_code, _ = generate_and_save_predicates(
        target_column, parsed["predicates"], parsed["composite_predicates"]
    )

    # Build the LTN / Python-lambda rule files directly from the parsed JSON,
    # skipping the LLM round-trip that natural-language rules go through.
    rule_texts = [canonical.rule_to_text(r) for r in parsed["rules"]]
    rules = [{"text": text, "structured": r}
             for text, r in zip(rule_texts, parsed["rules"])]
    save_rules_artifacts(parsed["predicates"], parsed["composite_predicates"], rules)

    with open(f"{OUTPUT_DIR}/ltn_rules.txt") as f:
        ltn_code = f.read()
    with open(f"{OUTPUT_DIR}/rules_only_rules.txt") as f:
        rules_code = f.read()

    return {
        **parsed,
        "rule_texts": rule_texts,
        "ltn_code": ltn_code,
        "rules_code": rules_code,
        "predicate_code": predicate_code,
        "lambda_code": lambda_code,
    }


def import_from_prolog(file_bytes: bytes, target_column: str) -> dict:
    """Import predicates + composite predicates + rules from a Prolog (.pl) file."""
    parsed = prolog_importer.import_prolog_source(file_bytes.decode("utf-8"))
    return _finalize_import(parsed, target_column)


def import_from_datalog(file_bytes: bytes, target_column: str) -> dict:
    """Import from a Datalog (.dl) file."""
    parsed = datalog_importer.import_datalog_source(file_bytes.decode("utf-8"))
    return _finalize_import(parsed, target_column)


def import_from_swrl(file_bytes: bytes, target_column: str) -> dict:
    """Import from a SWRL (.swrl) file (conjunctive rule subset)."""
    parsed = swrl_importer.import_swrl_source(file_bytes.decode("utf-8"))
    return _finalize_import(parsed, target_column)


def import_from_clips(file_bytes: bytes, target_column: str) -> dict:
    """Import from a CLIPS / Jess (.clp) file."""
    parsed = clips_importer.import_clips_source(file_bytes.decode("utf-8"))
    return _finalize_import(parsed, target_column)


def import_from_drools(file_bytes: bytes, target_column: str) -> dict:
    """Import from a Drools DRL (.drl) file."""
    parsed = drools_importer.import_drools_source(file_bytes.decode("utf-8"))
    return _finalize_import(parsed, target_column)


def import_from_decision_tree(file_bytes: bytes, target_column: str) -> dict:
    """Import from a pickled / joblib-dumped scikit-learn decision tree.

    The positive class defaults to the estimator's last class label; rules are
    generated for every leaf whose majority class is that positive class.
    """
    from utils import to_snake_case
    parsed = tree_importer.import_tree_obj(
        file_bytes, target_name=to_snake_case(target_column)
    )
    return _finalize_import(parsed, target_column)


# Map an uploaded file extension to its importer, for the multi-format UI uploader.
IMPORTERS_BY_EXT = {
    "pl": import_from_prolog,
    "dl": import_from_datalog,
    "swrl": import_from_swrl,
    "clp": import_from_clips,
    "drl": import_from_drools,
    "pkl": import_from_decision_tree,
    "joblib": import_from_decision_tree,
}
