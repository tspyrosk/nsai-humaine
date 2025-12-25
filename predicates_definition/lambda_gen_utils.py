from utils import to_snake_case

def generate_python_code(target_column, predicates, composite_predicates):
    code = ["def And(a, b): return a and b", "def Or(a, b): return a or b", "def Not(a): return not a"]
    
    # Generate normalized values
    for pred in predicates:
        norm_var = f"normalized_{to_snake_case(pred['name'])}"
        code.append(f"{norm_var} = normalize_value(X_train, {pred['column_index']}, {pred['threshold']})")
    
    code.append("")
    
    # Generate simple predicates
    for pred in predicates:
        snake_name = to_snake_case(pred['name'])
        comparison = {
            "Greater": ">",
            "Less": "<",
            "Equal": "=="
        }[pred['comparison']]
        code.append(f"{snake_name} = lambda x: x[{pred['column_index']}] {comparison} normalized_{snake_name}")
    
    code.append("")
    
    # Generate global declarations
    if target_column:
        target_snake = to_snake_case(target_column)
        code.append(f"global {target_snake}")
        code.append(f"global no_{target_snake}")
    for pred in composite_predicates:
        code.append(f"global {to_snake_case(pred['name'])}")
    
    code.append("")
    
    # Generate target predicates
    if target_column:
        to_snake_case(target_column)
        code.append(f"{target_snake} = lambda x, y: 1")
        code.append(f"no_{target_snake} = lambda x, y: 0")
    
    code.append("")
    
    # Generate composite predicates
    for pred in composite_predicates:
        expr = pred['expression']
        for p in predicates:
            expr = expr.replace(p['name'], f"{to_snake_case(p['name'])}(x)")
        code.append(f"{to_snake_case(pred['name'])} = lambda x: {expr}")

    code.append("")
    
    predicate_names = [to_snake_case(pred['name']) for pred in predicates]
    composite_predicate_names = [to_snake_case(pred['name']) for pred in composite_predicates]
    predicate_names.extend(composite_predicate_names)
    code.append(f"lambda_vars = [{",".join(predicate_names)}]")

    return "\n".join(code)