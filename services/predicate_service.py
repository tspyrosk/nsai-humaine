"""
Predicate Service - Handles predicate and rule generation and management.
"""
import os
from predicates_definition.predicate_gen_utils import generate_python_code
import predicates_definition.lambda_gen_utils as lambda_gen_utils
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
