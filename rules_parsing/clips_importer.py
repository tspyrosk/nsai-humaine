"""Import an NSAI rule set from a CLIPS / Jess (.clp) file.

CLIPS is the classic forward-chaining expert-system language; rules are
``defrule`` constructs and everything is written as S-expressions. The accepted
convention (see ``tests/fixtures/example_rules.clp``):

    (predicate Name ColumnIndex Threshold Comparison)
    (composite Name (and ...))                  ; and/or/not over names
    (defrule Name <CE> ... => (assert (Target ?x)))

Conditional elements (CEs) in a rule are ANDed together; ``(and ...)``,
``(or ...)`` and ``(not ...)`` group them explicitly. A predicate atom is a
unary pattern such as ``(clump_thickness ?x)``.
"""

from lark import Lark, Transformer, v_args

from rules_parsing import canonical


_GRAMMAR = r"""
    start: form+
    form: "(" item* ")"
    ?item: form | NUMBER | VAR | ARROW | SYMBOL

    ARROW: "=>"
    VAR: /\?[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: SIGNED_NUMBER
    SYMBOL: /[a-zA-Z_][a-zA-Z0-9_\-]*/

    COMMENT: ";" /[^\n]*/
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
    %ignore COMMENT
"""

# Sentinel emitted for the ``=>`` separator inside a defrule form.
_ARROW = object()


@v_args(inline=True)
class _SExpr(Transformer):
    def NUMBER(self, tok):
        s = str(tok)
        return float(s) if "." in s or "e" in s.lower() else int(s)

    def VAR(self, tok):
        return {"_kind": "var", "name": str(tok)[1:]}

    def ARROW(self, tok):
        return _ARROW

    def SYMBOL(self, tok):
        return str(tok)

    def form(self, *items):
        return list(items)

    def start(self, *forms):
        return list(forms)


_PARSER = Lark(_GRAMMAR, parser="lalr", transformer=_SExpr())


# ── S-expression → shared intermediate nodes ────────────────────────────────

def _expr_to_node(expr):
    """Convert a composite-expression S-expr (``(and a b)`` / leaf name) to a node."""
    if isinstance(expr, str):
        return canonical.compound(expr)
    head, *rest = expr
    return canonical.compound(head, [_expr_to_node(e) for e in rest])


def _ce_to_node(ce):
    """Convert a rule conditional element to a body node."""
    if not isinstance(ce, list):
        raise ValueError(f"Unsupported conditional element: {ce!r}")
    head, *rest = ce
    low = head.lower() if isinstance(head, str) else head
    if low == "and":
        return canonical.fold_and([_ce_to_node(c) for c in rest])
    if low == "or":
        return canonical.fold_or([_ce_to_node(c) for c in rest])
    if low == "not":
        if len(rest) != 1:
            raise ValueError("(not ...) takes exactly one conditional element")
        return canonical.not_node(_ce_to_node(rest[0]))
    # Otherwise a unary predicate pattern like (clump_thickness ?x); drop args.
    return canonical.compound(head)


def _defrule_to_clause(form):
    body_items = form[2:]  # skip 'defrule' and the rule name
    try:
        split = body_items.index(_ARROW)
    except ValueError:
        raise ValueError("defrule is missing the '=>' separator")
    ces, actions = body_items[:split], body_items[split + 1:]
    if not ces:
        raise ValueError("defrule has no conditional elements")
    if not actions:
        raise ValueError("defrule has no actions")

    body = canonical.fold_and([_ce_to_node(ce) for ce in ces])

    # Consequent: (assert (Target ?x)) — pull the asserted pattern's head.
    action = actions[0]
    asserted = action[1] if isinstance(action, list) and action[0] == "assert" else action
    head_name = asserted[0] if isinstance(asserted, list) else asserted
    return {"_kind": "rule", "head": canonical.compound(head_name), "body": body}


def _form_to_clause(form):
    head = form[0]
    if head == "predicate":
        name, col, thr, cmp = form[1], form[2], form[3], form[4]
        comp = canonical.compound("predicate", [
            canonical.compound(name), col, thr, canonical.compound(cmp),
        ])
        return {"_kind": "fact", "compound": comp}
    if head == "composite":
        name, expr = form[1], form[2]
        comp = canonical.compound("composite", [canonical.compound(name), _expr_to_node(expr)])
        return {"_kind": "fact", "compound": comp}
    if head == "defrule":
        return _defrule_to_clause(form)
    raise ValueError(f"Unknown top-level form: {head}")


def import_clips_source(source):
    forms = _PARSER.parse(source)
    clauses = [_form_to_clause(f) for f in forms]
    return canonical.clauses_to_canonical(clauses)


def import_clips_file(path):
    with open(path) as f:
        return import_clips_source(f.read())
