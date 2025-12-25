from utils import to_snake_case

def generate_python_code(target_column, predicates, composite_predicates):
    code = ["import tensorflow as tf", "import ltn", ""]
    
    # Generate normalized values
    for pred in predicates:
        norm_var = f"normalized_{to_snake_case(pred['name'])}"
        code.append(f"{norm_var} = normalize_value(X_train, {pred['column_index']}, {pred['threshold']})")
    
    code.append("")
    
    # Generate simple predicates
    for pred in predicates:
        snake_name = to_snake_case(pred['name'])
        comparison = {
            "Greater": "tf.math.greater",
            "Less": "tf.math.less",
            "Equal": "tf.math.equal"
        }[pred['comparison']]
        code.append(f"{snake_name} = ltn.Predicate(lambda x: {comparison}(x[:, {pred['column_index']}], normalized_{snake_name}))")
    
    code.append("")
    
    # Generate global declarations
    if target_column:
        target_snake = to_snake_case(target_column)
        code.append(f"global {target_snake}")
        code.append(f"global no_{target_snake}")
        code.append(f"global target")
        code.append(f"global no_target")
    for pred in composite_predicates:
        code.append(f"global {to_snake_case(pred['name'])}")
    
    code.append("")
    
    # Generate target predicates
    if target_column:
        target_snake = to_snake_case(target_column)
        code.append(f"{target_snake} = lambda x, y: f(x)")
        code.append(f"no_{target_snake} = lambda x, y: Not(f(x))")
        code.append(f"target = {target_snake}")
        code.append(f"no_target = no_{target_snake}")
    
    code.append("")
    
    # Generate composite predicates
    for pred in composite_predicates:
        expr = pred['expression']
        for p in predicates:
            expr = expr.replace(p['name'], f"{to_snake_case(p['name'])}(x)")
        code.append(f"{to_snake_case(pred['name'])} = lambda x: {expr}")
    
    return "\n".join(code)