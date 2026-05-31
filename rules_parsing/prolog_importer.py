"""Import an NSAI rule set from a Prolog (.pl) file.

The accepted Prolog subset is documented at the top of
``tests/fixtures/example_rules.pl``. Returns a dict with the same shape the
Streamlit UI keeps in ``session_state``:

    {
        "predicates":           [ {name, column_index, threshold, comparison, is_boolean}, ... ],
        "composite_predicates": [ {name, expression}, ... ],
        "rules":                [ {if_part, then_part}, ... ],
    }

``rules`` follow the pydantic ``Rule`` schema in ``rules_parsing/text2rules-v2.py``.
"""

from lark import Lark, Transformer, v_args


_GRAMMAR = r"""
    start: clause+

    clause: compound (":-" body)? "."

    ?body: disjunction
    ?disjunction: conjunction
                | disjunction ";" conjunction   -> or_expr
    ?conjunction: unary
                | conjunction "," unary          -> and_expr
    ?unary: "\\+" unary                          -> not_expr
          | "(" disjunction ")"
          | compound

    compound: CNAME ("(" args ")")?
    args: arg ("," arg)*
    ?arg: compound | number

    number: SIGNED_NUMBER

    COMMENT: "%" /[^\n]*/
    %import common.CNAME
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
    %ignore COMMENT
"""


_COMPARISON_MAP = {"greater": "Greater", "less": "Less", "equal": "Equal"}
_COMPOSITE_OP_MAP = {"and": "And", "or": "Or", "not": "Not"}


@v_args(inline=True)
class _Tree(Transformer):
    """Lower the parse tree to plain Python structures."""

    def number(self, tok):
        s = str(tok)
        return float(s) if "." in s or "e" in s.lower() else int(s)

    def args(self, *xs):
        return list(xs)

    def compound(self, name, args=None):
        return {"_kind": "compound", "name": str(name), "args": args or []}

    def not_expr(self, inner):
        return {"_kind": "not", "arg": inner}

    def and_expr(self, left, right):
        return {"_kind": "and", "left": left, "right": right}

    def or_expr(self, left, right):
        return {"_kind": "or", "left": left, "right": right}

    def clause(self, compound, body=None):
        if body is None:
            return {"_kind": "fact", "compound": compound}
        return {"_kind": "rule", "head": compound, "body": body}

    def start(self, *clauses):
        return list(clauses)


_PARSER = Lark(_GRAMMAR, parser="lalr", transformer=_Tree())


def _compound_name(node):
    if node["_kind"] != "compound":
        raise ValueError(f"Expected compound, got {node['_kind']}")
    return node["name"]


def _composite_expression_to_string(node):
    """Render a composite-predicate compound back to the title-case string the UI uses."""
    if node["_kind"] != "compound":
        raise ValueError(f"Unsupported composite node: {node}")
    if not node["args"]:
        # Bare atom — leaf predicate name.
        return node["name"]
    op = _COMPOSITE_OP_MAP.get(node["name"])
    if op is None:
        raise ValueError(f"Unsupported composite operator: {node['name']}")
    rendered = ", ".join(_composite_expression_to_string(a) for a in node["args"])
    return f"{op}({rendered})"


def _body_to_rule_part(node):
    """Convert a rule-body AST node into the Predicate/Binary/Unary dicts the UI uses."""
    kind = node["_kind"]
    if kind == "compound":
        # Predicate call P(X) — we drop the variable argument; the UI only cares about the name.
        return {"name": node["name"]}
    if kind == "not":
        inner = node["arg"]
        if inner["_kind"] != "compound":
            raise ValueError("NOT (\\+) is only supported over a single predicate call")
        return {"operator": "NOT", "name": inner["name"]}
    if kind in ("and", "or"):
        return {
            "operator": "AND" if kind == "and" else "OR",
            "arg1": _body_to_rule_part(node["left"]),
            "arg2": _body_to_rule_part(node["right"]),
        }
    raise ValueError(f"Unsupported body node: {node}")


def _parse_predicate_fact(compound):
    """predicate(Name, ColIdx, Threshold, Comparison) → dict."""
    name_node, column_index, threshold, comparison_node = compound["args"]
    comp_atom = _compound_name(comparison_node)
    if comp_atom not in _COMPARISON_MAP:
        raise ValueError(f"Unknown comparison atom: {comp_atom!r}")
    return {
        "name": _compound_name(name_node),
        "column_index": int(column_index),
        "threshold": float(threshold),
        "comparison": _COMPARISON_MAP[comp_atom],
        "is_boolean": False,
    }


def _parse_composite_fact(compound):
    """composite(Name, Expr) → dict."""
    name_node, expr = compound["args"]
    return {
        "name": _compound_name(name_node),
        "expression": _composite_expression_to_string(expr),
    }


def import_prolog_file(path):
    with open(path) as f:
        source = f.read()
    return import_prolog_source(source)


def rule_to_text(rule):
    """Render a parsed rule dict as a readable ``IF ... THEN ...`` string for UI display."""
    return f"IF {_expr_to_text(rule['if_part'])} THEN {_unary_to_text(rule['then_part'])}"


def _unary_to_text(node):
    return f"NOT {node['name']}" if node.get("operator") == "NOT" else node["name"]


def _expr_to_text(node):
    op = node.get("operator")
    if op in ("AND", "OR"):
        return f"({_expr_to_text(node['arg1'])} {op} {_expr_to_text(node['arg2'])})"
    return _unary_to_text(node)


def rule_to_ltn(rule):
    """Render a parsed rule into the LTN expression text used in ``ltn_rules.txt``."""
    return f"Forall(x, Implies({_to_ltn(rule['if_part'])}, {_unary_to_ltn(rule['then_part'])}))"


def _to_ltn(node):
    op = node.get("operator")
    if op == "AND":
        return f"And({_to_ltn(node['arg1'])}, {_to_ltn(node['arg2'])})"
    if op == "OR":
        return f"Or({_to_ltn(node['arg1'])}, {_to_ltn(node['arg2'])})"
    if op == "NOT":
        return f"Not({node['name']}(x))"
    return f"{node['name']}(x)"


def _unary_to_ltn(node):
    if node.get("operator") == "NOT":
        return f"Not({node['name']}(x,f(x)))"
    return f"{node['name']}(x,f(x))"


def rule_to_python_lambda(rule):
    """Render a parsed rule into the Python-lambda tuple used in ``rules_only_rules.txt``."""
    return f"({_to_py(rule['if_part'])}, {_unary_to_py(rule['then_part'])})"


def _to_py(node):
    op = node.get("operator")
    if op == "AND":
        return f"({_to_py(node['arg1'])} and {_to_py(node['arg2'])})"
    if op == "OR":
        return f"({_to_py(node['arg1'])} or {_to_py(node['arg2'])})"
    if op == "NOT":
        return f"not({node['name']}(x))"
    return f"{node['name']}(x)"


def _unary_to_py(node):
    if node.get("operator") == "NOT":
        return f"not({node['name']}(x,y))"
    return f"{node['name']}(x,y)"


def import_prolog_source(source):
    clauses = _PARSER.parse(source)

    predicates = []
    composite_predicates = []
    rules = []

    for clause in clauses:
        if clause["_kind"] == "fact":
            compound = clause["compound"]
            head = compound["name"]
            if head == "predicate":
                predicates.append(_parse_predicate_fact(compound))
            elif head == "composite":
                composite_predicates.append(_parse_composite_fact(compound))
            else:
                raise ValueError(f"Unknown fact head: {head}")
        elif clause["_kind"] == "rule":
            rules.append({
                "if_part": _body_to_rule_part(clause["body"]),
                "then_part": {"name": clause["head"]["name"]},
            })
        else:
            raise ValueError(f"Unexpected clause: {clause}")

    return {
        "predicates": predicates,
        "composite_predicates": composite_predicates,
        "rules": rules,
    }
