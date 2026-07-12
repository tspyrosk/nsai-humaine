"""Import an NSAI rule set from a SWRL (.swrl) file.

SWRL (Semantic Web Rule Language) writes rules in human-readable syntax as a
conjunction of atoms implying a consequent::

    atom ^ atom ^ ... -> atom

Core SWRL antecedents are conjunction-only — there is no disjunction or
negation — so this importer accepts the conjunctive subset of the rule model
(documented in ``tests/fixtures/example_rules.swrl``). Predicate and composite
metadata, which SWRL has no native construct for, is carried by two reserved
declaration atoms written without a consequent:

    Predicate(Name, ColumnIndex, Threshold, Comparison)
    Composite(Name, Expression)          % Expression: and(...)/or(...)/not(...)

Rule atoms are unary predicate calls over a variable, e.g. ``clump_thickness(?x)``.
"""

from lark import Lark, Transformer, v_args

from rules_parsing import canonical


_GRAMMAR = r"""
    start: statement+

    statement: conj ("->" compound)?

    conj: compound ("^" compound)*

    compound: CNAME ("(" args ")")?
    args: arg ("," arg)*
    ?arg: compound | number | VAR

    number: SIGNED_NUMBER
    VAR: /\?[a-zA-Z_][a-zA-Z0-9_]*/

    COMMENT: "#" /[^\n]*/
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

    def VAR(self, tok):
        # Variables only appear as rule-atom arguments, which downstream lowering
        # discards; keep a lightweight marker so arg lists stay uniform.
        return {"_kind": "var", "name": str(tok)[1:]}

    def args(self, *xs):
        return list(xs)

    def compound(self, name, args=None):
        return canonical.compound(name, args or [])

    def conj(self, *atoms):
        return list(atoms)

    def statement(self, conj, head=None):
        if head is None:
            # Declaration atom (Predicate/Composite) — a single-atom conjunction.
            if len(conj) != 1:
                raise ValueError("A statement without '->' must be a single declaration atom")
            return {"_kind": "fact", "compound": conj[0]}
        return {"_kind": "rule", "head": head, "body": canonical.fold_and(conj)}

    def start(self, *statements):
        return list(statements)


_PARSER = Lark(_GRAMMAR, parser="lalr", transformer=_Tree())


def import_swrl_source(source):
    return canonical.clauses_to_canonical(_PARSER.parse(source))


def import_swrl_file(path):
    with open(path) as f:
        return import_swrl_source(f.read())
