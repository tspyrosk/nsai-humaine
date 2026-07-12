"""Import an NSAI rule set from a Prolog (.pl) file.

The accepted Prolog subset is documented at the top of
``tests/fixtures/example_rules.pl``. Returns the canonical importer dict
described in ``rules_parsing/canonical.py``.

Format-agnostic lowering and rendering live in ``canonical``; this module only
owns the Prolog grammar and the Transformer that emits the shared
intermediate-node shapes.
"""

from lark import Lark, Transformer, v_args

from rules_parsing import canonical
# Re-exported for backwards compatibility (callers historically imported these
# rendering helpers from prolog_importer).
from rules_parsing.canonical import (  # noqa: F401
    rule_to_text,
    rule_to_ltn,
    rule_to_python_lambda,
)


_GRAMMAR = r"""
    start: clause+

    clause: head (":-" body)? "."

    ?head: compound
         | "\\+" compound                        -> neg_head

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


@v_args(inline=True)
class _Tree(Transformer):
    """Lower the parse tree to the shared intermediate-node shapes."""

    def number(self, tok):
        s = str(tok)
        return float(s) if "." in s or "e" in s.lower() else int(s)

    def args(self, *xs):
        return list(xs)

    def compound(self, name, args=None):
        return canonical.compound(name, args or [])

    def not_expr(self, inner):
        return canonical.not_node(inner)

    def and_expr(self, left, right):
        return canonical.and_node(left, right)

    def or_expr(self, left, right):
        return canonical.or_node(left, right)

    def neg_head(self, comp):
        return canonical.not_node(comp)

    def clause(self, head, body=None):
        if body is None:
            # A bodiless clause is a fact declaration; a negated head is meaningless there.
            if head.get("_kind") == "not":
                raise ValueError("A negated head (\\+) requires a rule body")
            return {"_kind": "fact", "compound": head}
        return {"_kind": "rule", "head": head, "body": body}

    def start(self, *clauses):
        return list(clauses)


_PARSER = Lark(_GRAMMAR, parser="lalr", transformer=_Tree())


def import_prolog_source(source):
    return canonical.clauses_to_canonical(_PARSER.parse(source))


def import_prolog_file(path):
    with open(path) as f:
        return import_prolog_source(f.read())
