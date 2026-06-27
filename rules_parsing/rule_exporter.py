"""Export a canonical rule set back into a serialized rule format.

This is the inverse of the importers in this package: it takes the canonical
dict ``{predicates, composite_predicates, rules}`` (see ``canonical``) and
renders it as Prolog / Datalog / SWRL / CLIPS / Drools text, or canonical JSON.
The text formats are rendered to the same conventions the matching importers
accept, so ``export → import`` round-trips back to the original dict (modulo
SWRL, which is conjunction-only).

Used when publishing a model so the symbolic rules travel with it.
"""

import json

from rules_parsing import canonical


# ext -> human label, in the order the UI selector should present them.
SUPPORTED_FORMATS = {
    "pl": "Prolog (.pl)",
    "dl": "Datalog (.dl)",
    "swrl": "SWRL (.swrl)",
    "clp": "CLIPS (.clp)",
    "drl": "Drools DRL (.drl)",
    "json": "Canonical JSON (.json)",
}


def export_rule_set(rule_set, fmt):
    """Render the canonical ``rule_set`` dict to the format ``fmt`` (an extension)."""
    fmt = fmt.lower().lstrip(".")
    try:
        renderer = _RENDERERS[fmt]
    except KeyError:
        raise ValueError(f"Unsupported export format: {fmt!r}")
    return renderer(rule_set)


def _fmt_num(n):
    """Render a threshold/column number without scientific notation."""
    if isinstance(n, bool):
        return str(int(n))
    if isinstance(n, int):
        return str(n)
    return repr(float(n))


# ── Prolog / Datalog ─────────────────────────────────────────────────────────

def _logic_render(rule_set, *, not_kw):
    """Shared Prolog/Datalog renderer; ``not_kw`` is ``\\+`` (pl) or ``not`` (dl)."""
    lines = []
    for p in rule_set.get("predicates", []):
        comp = canonical.REVERSE_COMPARISON_MAP[p["comparison"]]
        lines.append(
            f"predicate({p['name']}, {p['column_index']}, "
            f"{_fmt_num(p['threshold'])}, {comp})."
        )
    for c in rule_set.get("composite_predicates", []):
        node = canonical.parse_composite_expression(c["expression"])
        lines.append(f"composite({c['name']}, {_functor_expr(node)}).")
    for rule in rule_set.get("rules", []):
        head = rule["then_part"]["name"]
        body = _logic_body(rule["if_part"], not_kw=not_kw)
        lines.append(f"{head}(X) :- {body}.")
    return "\n".join(lines) + "\n"


def _functor_expr(node):
    """Render a composite node in functor style: and(a, b) / or(a, b) / not(a)."""
    if not node["args"]:
        return node["name"]
    inner = ", ".join(_functor_expr(a) for a in node["args"])
    return f"{node['name'].lower()}({inner})"


def _logic_body(node, *, not_kw):
    op = node.get("operator")
    if op == "AND":
        return f"({_logic_body(node['arg1'], not_kw=not_kw)}, {_logic_body(node['arg2'], not_kw=not_kw)})"
    if op == "OR":
        return f"({_logic_body(node['arg1'], not_kw=not_kw)} ; {_logic_body(node['arg2'], not_kw=not_kw)})"
    if op == "NOT":
        return f"{not_kw} {node['name']}(X)"
    return f"{node['name']}(X)"


def _render_prolog(rule_set):
    return "% NSAI rule export (Prolog)\n" + _logic_render(rule_set, not_kw="\\+")


def _render_datalog(rule_set):
    return "% NSAI rule export (Datalog)\n" + _logic_render(rule_set, not_kw="not")


# ── CLIPS ────────────────────────────────────────────────────────────────────

def _render_clips(rule_set):
    lines = ["; NSAI rule export (CLIPS)"]
    for p in rule_set.get("predicates", []):
        comp = canonical.REVERSE_COMPARISON_MAP[p["comparison"]]
        lines.append(
            f"(predicate {p['name']} {p['column_index']} "
            f"{_fmt_num(p['threshold'])} {comp})"
        )
    for c in rule_set.get("composite_predicates", []):
        node = canonical.parse_composite_expression(c["expression"])
        lines.append(f"(composite {c['name']} {_clips_expr(node)})")
    for i, rule in enumerate(rule_set.get("rules", []), start=1):
        head = rule["then_part"]["name"]
        ce = _clips_ce(rule["if_part"])
        lines.append(f"(defrule r{i} {ce} => (assert ({head} ?x)))")
    return "\n".join(lines) + "\n"


def _clips_expr(node):
    if not node["args"]:
        return node["name"]
    inner = " ".join(_clips_expr(a) for a in node["args"])
    return f"({node['name'].lower()} {inner})"


def _clips_ce(node):
    op = node.get("operator")
    if op == "AND":
        return f"(and {_clips_ce(node['arg1'])} {_clips_ce(node['arg2'])})"
    if op == "OR":
        return f"(or {_clips_ce(node['arg1'])} {_clips_ce(node['arg2'])})"
    if op == "NOT":
        return f"(not ({node['name']} ?x))"
    return f"({node['name']} ?x)"


# ── Drools DRL ───────────────────────────────────────────────────────────────

def _render_drools(rule_set):
    lines = ["// NSAI rule export (Drools DRL)"]
    for p in rule_set.get("predicates", []):
        comp = canonical.REVERSE_COMPARISON_MAP[p["comparison"]]
        lines.append(
            f"predicate( {p['name']}, {p['column_index']}, "
            f"{_fmt_num(p['threshold'])}, {comp} )"
        )
    for c in rule_set.get("composite_predicates", []):
        node = canonical.parse_composite_expression(c["expression"])
        lines.append(f"composite( {c['name']}, {_functor_expr(node)} )")
    for i, rule in enumerate(rule_set.get("rules", []), start=1):
        head = rule["then_part"]["name"]
        lhs = _drools_lhs(rule["if_part"])
        lines.append(f'rule "r{i}"\nwhen\n    {lhs}\nthen\n    {head}();\nend')
    return "\n".join(lines) + "\n"


def _drools_lhs(node):
    op = node.get("operator")
    if op == "AND":
        return f"{_drools_wrap(node['arg1'])} and {_drools_wrap(node['arg2'])}"
    if op == "OR":
        return f"{_drools_wrap(node['arg1'])} or {_drools_wrap(node['arg2'])}"
    if op == "NOT":
        return f"not {node['name']}()"
    return f"{node['name']}()"


def _drools_wrap(node):
    """Parenthesize a binary child so the explicit rule structure is preserved."""
    if node.get("operator") in ("AND", "OR"):
        return f"( {_drools_lhs(node)} )"
    return _drools_lhs(node)


# ── SWRL (conjunction-only) ──────────────────────────────────────────────────

def _render_swrl(rule_set):
    lines = ["# NSAI rule export (SWRL — conjunctive rules only)"]
    for p in rule_set.get("predicates", []):
        comp = canonical.REVERSE_COMPARISON_MAP[p["comparison"]]
        lines.append(
            f"Predicate({p['name']}, {p['column_index']}, "
            f"{_fmt_num(p['threshold'])}, {comp})"
        )
    for c in rule_set.get("composite_predicates", []):
        node = canonical.parse_composite_expression(c["expression"])
        lines.append(f"Composite({c['name']}, {_functor_expr(node)})")
    for rule in rule_set.get("rules", []):
        leaves = _conjunctive_leaves(rule["if_part"])
        if leaves is None:
            # SWRL cannot express OR/NOT in the antecedent — skip with a note.
            lines.append(f"# skipped non-conjunctive rule: {canonical.rule_to_text(rule)}")
            continue
        head = rule["then_part"]["name"]
        antecedent = " ^ ".join(f"{name}(?x)" for name in leaves)
        lines.append(f"{antecedent} -> {head}(?x)")
    return "\n".join(lines) + "\n"


def _conjunctive_leaves(node):
    """Flatten a pure-AND tree of plain predicates to a name list, else ``None``."""
    op = node.get("operator")
    if op == "AND":
        left = _conjunctive_leaves(node["arg1"])
        right = _conjunctive_leaves(node["arg2"])
        if left is None or right is None:
            return None
        return left + right
    if op in ("OR", "NOT"):
        return None
    return [node["name"]]


# ── JSON ─────────────────────────────────────────────────────────────────────

def _render_json(rule_set):
    return json.dumps(rule_set, indent=2) + "\n"


_RENDERERS = {
    "pl": _render_prolog,
    "dl": _render_datalog,
    "swrl": _render_swrl,
    "clp": _render_clips,
    "drl": _render_drools,
    "json": _render_json,
}
