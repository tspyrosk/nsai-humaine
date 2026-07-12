"""Import an NSAI rule set from a Datalog (.dl) file.

Datalog is the function-free, rule-based subset of Prolog used by deductive
databases. The accepted convention mirrors the Prolog one (see
``tests/fixtures/example_rules.dl``):

    predicate(Name, ColumnIndex, Threshold, Comparison).
    composite(Name, Expression).        % Expression: and/2, or/2, not/1
    target(X) :- Body.

Body connectives:
    ,         conjunction (AND)
    ;         disjunction (OR)
    not / \\+  negation (NOT)

Compared with the Prolog importer the only grammatical difference is that
Datalog spells negation with the ``not`` keyword (``\\+`` is also accepted for
convenience). Everything else — lowering, rendering — is shared via
``canonical``.
"""

from lark import Lark, Transformer, v_args

from rules_parsing import canonical


_GRAMMAR = r"""
    start: clause+

    clause: head (":-" body)? "."

    ?head: compound
         | ("not"|"\\+") compound                -> neg_head

    ?body: disjunction
    ?disjunction: conjunction
                | disjunction ";" conjunction   -> or_expr
    ?conjunction: unary
                | conjunction "," unary          -> and_expr
    ?unary: ("not"|"\\+") unary                  -> not_expr
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
            if head.get("_kind") == "not":
                raise ValueError("A negated head requires a rule body")
            return {"_kind": "fact", "compound": head}
        return {"_kind": "rule", "head": head, "body": body}

    def start(self, *clauses):
        return list(clauses)


# ``not`` is a keyword here, so it must win over the CNAME terminal; the Earley
# parser handles the keyword/identifier overlap without extra priorities.
_PARSER = Lark(_GRAMMAR, parser="lalr", transformer=_Tree())


def import_datalog_source(source):
    return canonical.clauses_to_canonical(_PARSER.parse(source))


def import_datalog_file(path):
    with open(path) as f:
        return import_datalog_source(f.read())
