"""
Predicate Service - Handles predicate and rule generation and management.
"""
import os
import json
from predicates_definition.predicate_gen_utils import generate_python_code
import predicates_definition.lambda_gen_utils as lambda_gen_utils
from rules_parsing import (
    canonical,
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


def save_and_parse_rules(selected_rules: list, predicates: list = None,
                         composite_predicates: list = None) -> tuple:
    """
    Save rules and parse them into executable code.

    Args:
        selected_rules: List of natural-language rule strings
        predicates: Simple predicate definitions (for persisting rules.json)
        composite_predicates: Composite predicate definitions

    Returns:
        Tuple of (ltn_code, rules_code)
    """
    # Save raw rules
    with open(f"{OUTPUT_DIR}/rules_raw.txt", "w") as text_file:
        text_file.write("\n".join(selected_rules))

    # Parse rules
    os.system(f"python {TEXT2RULES_SCRIPT} --raw_rules_path={OUTPUT_DIR}/rules_raw.txt --output_path={OUTPUT_DIR}")

    # Read generated code
    with open(f"{OUTPUT_DIR}/rules_only_rules.txt", "r") as file:
        rules_code = file.read()

    with open(f"{OUTPUT_DIR}/ltn_rules.txt", "r") as file:
        ltn_code = file.read()

    # Persist the canonical rule set from the structured rules text2rules wrote,
    # combined with the predicates/composites defined in this session.
    structured_path = f"{OUTPUT_DIR}/rules_structured.json"
    if os.path.exists(structured_path):
        with open(structured_path) as f:
            structured_rules = json.load(f)
        save_rule_set(predicates or [], composite_predicates or [], structured_rules)

    return ltn_code, rules_code


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
    ltn_rule_strs = [canonical.rule_to_ltn(r) for r in parsed["rules"]]
    py_rule_strs = [canonical.rule_to_python_lambda(r) for r in parsed["rules"]]
    rule_texts = [canonical.rule_to_text(r) for r in parsed["rules"]]

    ltn_code = f"parsed_rules = lambda x, y: [{', '.join(ltn_rule_strs)}]"
    rules_code = f"parsed_rules_python = lambda x, y: [{', '.join(py_rule_strs)}]"

    with open(f"{OUTPUT_DIR}/ltn_rules.txt", "w") as f:
        f.write(ltn_code)
    with open(f"{OUTPUT_DIR}/rules_only_rules.txt", "w") as f:
        f.write(rules_code)
    with open(f"{OUTPUT_DIR}/rules_raw.txt", "w") as f:
        f.write("\n".join(rule_texts))

    # Persist the canonical rule set so the publish/export flow can serialize it.
    save_rule_set(parsed["predicates"], parsed["composite_predicates"], parsed["rules"])

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
