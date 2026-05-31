"""
Predicate Service - Handles predicate and rule generation and management.
"""
import os
from predicates_definition.predicate_gen_utils import generate_python_code
import predicates_definition.lambda_gen_utils as lambda_gen_utils
from rules_parsing import prolog_importer
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


def save_and_parse_rules(selected_rules: list) -> tuple:
    """
    Save rules and parse them into executable code.

    Args:
        selected_rules: List of rule strings

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

    return ltn_code, rules_code


def import_from_prolog(file_bytes: bytes, target_column: str) -> dict:
    """Import predicates + composite predicates + rules from a Prolog (.pl) file.

    Populates the same output artifacts (predicates.txt, lambdas.txt,
    predicate_names.txt, ltn_rules.txt, rules_only_rules.txt) that the
    interactive UI flow produces, so training can start immediately.

    Returns the parsed dict augmented with ``rule_texts`` (human-readable
    ``IF ... THEN ...`` strings for display) and ``ltn_code`` / ``rules_code``
    (generated source the UI surfaces back to the user).
    """
    from utils import to_snake_case

    source = file_bytes.decode("utf-8")
    parsed = prolog_importer.import_prolog_source(source)

    # Rule heads in the .pl file (e.g. `target(X) :- ...`) need to resolve to
    # the lambda that lambda_gen_utils.py writes under the snake-cased target
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
    ltn_rule_strs = [prolog_importer.rule_to_ltn(r) for r in parsed["rules"]]
    py_rule_strs = [prolog_importer.rule_to_python_lambda(r) for r in parsed["rules"]]
    rule_texts = [prolog_importer.rule_to_text(r) for r in parsed["rules"]]

    ltn_code = f"parsed_rules = lambda x, y: [{', '.join(ltn_rule_strs)}]"
    rules_code = f"parsed_rules_python = lambda x, y: [{', '.join(py_rule_strs)}]"

    with open(f"{OUTPUT_DIR}/ltn_rules.txt", "w") as f:
        f.write(ltn_code)
    with open(f"{OUTPUT_DIR}/rules_only_rules.txt", "w") as f:
        f.write(rules_code)
    with open(f"{OUTPUT_DIR}/rules_raw.txt", "w") as f:
        f.write("\n".join(rule_texts))

    return {
        **parsed,
        "rule_texts": rule_texts,
        "ltn_code": ltn_code,
        "rules_code": rules_code,
        "predicate_code": predicate_code,
        "lambda_code": lambda_code,
    }
