"""Shared canonical-model helpers for every rule-format importer.

Each importer (Prolog, Datalog, SWRL, CLIPS, Drools, decision tree) lowers its
source into the same *clause* list and then into the canonical dict the
Streamlit UI keeps in ``session_state``:

    {
        "predicates":           [ {name, column_index, threshold, comparison, is_boolean}, ... ],
        "composite_predicates": [ {name, expression}, ... ],
        "rules":                [ {if_part, then_part}, ... ],
    }

``rules`` follow the pydantic ``Rule`` schema in ``rules_parsing/text2rules-v2.py``.

Intermediate node shapes (what each grammar's Transformer is expected to emit):

    compound: {"_kind": "compound", "name": str, "args": [node | number, ...]}
    not:      {"_kind": "not", "arg": node}
    and/or:   {"_kind": "and"|"or", "left": node, "right": node}

    clause (fact): {"_kind": "fact", "compound": <compound>}
    clause (rule): {"_kind": "rule", "head": <compound>, "body": <body node>}

By sharing this lowering + rendering code, a new format only has to provide a
grammar/Transformer (or, for decision trees, a tree walk) that produces the
clause list above — everything downstream is identical to Prolog.
"""

# Comparison atoms (declaration side) → the title-case strings the UI/codegen use.
COMPARISON_MAP = {"greater": "Greater", "less": "Less", "equal": "Equal"}
# Inverse of COMPARISON_MAP, used by exporters to render predicate facts.
REVERSE_COMPARISON_MAP = {v: k for k, v in COMPARISON_MAP.items()}
# Composite-expression functors → the title-case operator the UI renders.
COMPOSITE_OP_MAP = {"and": "And", "or": "Or", "not": "Not"}

# Heads that mark a fact as a predicate / composite declaration (case-insensitive).
PREDICATE_FACT_HEADS = {"predicate"}
COMPOSITE_FACT_HEADS = {"composite"}


# ── intermediate-node constructors (handy for hand-built importers) ──────────

def compound(name, args=None):
    return {"_kind": "compound", "name": str(name), "args": list(args or [])}


def not_node(inner):
    return {"_kind": "not", "arg": inner}


def and_node(left, right):
    return {"_kind": "and", "left": left, "right": right}


def or_node(left, right):
    return {"_kind": "or", "left": left, "right": right}


def fold_and(nodes):
    """Left-fold a list of body nodes into nested binary ``and`` nodes."""
    return _fold(nodes, and_node)


def fold_or(nodes):
    """Left-fold a list of body nodes into nested binary ``or`` nodes."""
    return _fold(nodes, or_node)


def _fold(nodes, op):
    nodes = list(nodes)
    if not nodes:
        raise ValueError("Cannot fold an empty node list")
    acc = nodes[0]
    for node in nodes[1:]:
        acc = op(acc, node)
    return acc


# ── lowering: intermediate nodes → canonical dict pieces ─────────────────────

def compound_name(node):
    if node["_kind"] != "compound":
        raise ValueError(f"Expected compound, got {node['_kind']}")
    return node["name"]


def composite_expression_to_string(node):
    """Render a composite-predicate compound back to the title-case string the UI uses."""
    if node["_kind"] != "compound":
        raise ValueError(f"Unsupported composite node: {node}")
    if not node["args"]:
        # Bare atom — leaf predicate name.
        return node["name"]
    op = COMPOSITE_OP_MAP.get(node["name"].lower())
    if op is None:
        raise ValueError(f"Unsupported composite operator: {node['name']}")
    rendered = ", ".join(composite_expression_to_string(a) for a in node["args"])
    return f"{op}({rendered})"


def parse_composite_expression(expr):
    """Inverse of ``composite_expression_to_string``: title-case string → node.

    Parses the ``And(a, b)`` / ``Or(...)`` / ``Not(a)`` / bare-leaf strings the
    UI stores in a composite's ``expression`` field back into a compound node so
    exporters can re-render them in each target syntax.
    """
    tokens = _tokenize_composite(expr)
    pos = 0

    def parse():
        nonlocal pos
        name = tokens[pos]
        pos += 1
        if pos < len(tokens) and tokens[pos] == "(":
            pos += 1  # consume '('
            args = [parse()]
            while tokens[pos] == ",":
                pos += 1  # consume ','
                args.append(parse())
            pos += 1  # consume ')'
            return compound(name, args)
        return compound(name)

    node = parse()
    if pos != len(tokens):
        raise ValueError(f"Trailing tokens in composite expression: {expr!r}")
    return node


def _tokenize_composite(expr):
    tokens = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
        elif ch in "(),":
            tokens.append(ch)
            i += 1
        else:
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            if j == i:
                raise ValueError(f"Unexpected character {ch!r} in composite expression: {expr!r}")
            tokens.append(expr[i:j])
            i = j
    return tokens


def body_to_rule_part(node):
    """Convert a rule-body node into the Predicate/Binary/Unary dicts the UI uses."""
    kind = node["_kind"]
    if kind == "compound":
        # Predicate call P(X) — we drop the variable argument; the UI only cares about the name.
        return {"name": node["name"]}
    if kind == "not":
        inner = node["arg"]
        if inner["_kind"] != "compound":
            raise ValueError("NOT is only supported over a single predicate call")
        return {"operator": "NOT", "name": inner["name"]}
    if kind in ("and", "or"):
        return {
            "operator": "AND" if kind == "and" else "OR",
            "arg1": body_to_rule_part(node["left"]),
            "arg2": body_to_rule_part(node["right"]),
        }
    raise ValueError(f"Unsupported body node: {node}")


def head_to_then_part(node):
    """Convert a rule-head node into the ``then_part`` dict the UI/codegen use.

    A plain compound head → ``{"name": ...}``. A NOT-wrapped compound head — an
    imported *negative goal*, e.g. Prolog ``\\+ target(X) :- Body`` — →
    ``{"operator": "NOT", "name": ...}``, so a rule can conclude the negative
    class (``Not(target(x,f(x)))`` in the LTN layer).
    """
    if node["_kind"] == "compound":
        return {"name": node["name"]}
    if node["_kind"] == "not":
        inner = node["arg"]
        if inner["_kind"] != "compound":
            raise ValueError("A negated rule head must wrap a single predicate call")
        return {"operator": "NOT", "name": inner["name"]}
    raise ValueError(f"Unsupported rule head: {node}")


def parse_predicate_fact(comp):
    """predicate(Name, ColIdx, Threshold, Comparison) → predicate dict."""
    name_node, column_index, threshold, comparison_node = comp["args"]
    comp_atom = compound_name(comparison_node).lower()
    if comp_atom not in COMPARISON_MAP:
        raise ValueError(f"Unknown comparison atom: {comp_atom!r}")
    return {
        "name": compound_name(name_node),
        "column_index": int(column_index),
        "threshold": float(threshold),
        "comparison": COMPARISON_MAP[comp_atom],
        "is_boolean": False,
    }


def parse_composite_fact(comp):
    """composite(Name, Expr) → composite dict."""
    name_node, expr = comp["args"]
    return {
        "name": compound_name(name_node),
        "expression": composite_expression_to_string(expr),
    }


def clauses_to_canonical(clauses):
    """Lower a clause list (facts + rules) into the canonical importer dict."""
    predicates = []
    composite_predicates = []
    rules = []

    for clause in clauses:
        if clause["_kind"] == "fact":
            comp = clause["compound"]
            head = comp["name"].lower()
            if head in PREDICATE_FACT_HEADS:
                predicates.append(parse_predicate_fact(comp))
            elif head in COMPOSITE_FACT_HEADS:
                composite_predicates.append(parse_composite_fact(comp))
            else:
                raise ValueError(f"Unknown fact head: {comp['name']}")
        elif clause["_kind"] == "rule":
            rules.append({
                "if_part": body_to_rule_part(clause["body"]),
                "then_part": head_to_then_part(clause["head"]),
            })
        else:
            raise ValueError(f"Unexpected clause: {clause}")

    return {
        "predicates": predicates,
        "composite_predicates": composite_predicates,
        "rules": rules,
    }


# ── rendering: canonical rule dict → the text/LTN/Python forms the app uses ───

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
