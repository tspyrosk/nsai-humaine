"""Import an NSAI rule set from a Drools DRL (.drl) file.

Drools is the de-facto open-source business-rules engine; rules use the
``rule "name" when <LHS> then <RHS> end`` form. The accepted convention (see
``tests/fixtures/example_rules.drl``):

    predicate( Name, ColumnIndex, Threshold, Comparison )
    composite( Name, and(...) )                 // and/or/not over names

    rule "r1"
    when
        clump_thickness() and mitoses()
    then
        target();
    end

Antecedents use the DRL connectives ``and`` / ``or`` / ``not`` with parentheses
for grouping; a predicate atom is a nullary pattern such as ``clump_thickness()``.
The consequent names the target predicate.
"""

from lark import Lark, Transformer, v_args

from rules_parsing import canonical


_GRAMMAR = r"""
    start: (decl | rule_def)+

    decl: "predicate" "(" CNAME "," number "," number "," CNAME ")"   -> pred_decl
        | "composite" "(" CNAME "," expr ")"                          -> comp_decl

    ?expr: "and" "(" expr "," expr ")"   -> and_c
         | "or" "(" expr "," expr ")"    -> or_c
         | "not" "(" expr ")"            -> not_c
         | CNAME                         -> leaf

    rule_def: "rule" ESCAPED_STRING "when" lhs "then" rhs "end"

    ?lhs: or_expr
    ?or_expr: and_expr
            | or_expr "or" and_expr      -> or_b
    ?and_expr: unary
             | and_expr "and" unary      -> and_b
    ?unary: "not" unary                  -> not_b
          | "(" or_expr ")"
          | atom

    atom: CNAME "(" ")"
    rhs: CNAME "(" ")" ";"?           -> pos_rhs
       | "not" CNAME "(" ")" ";"?     -> neg_rhs

    number: SIGNED_NUMBER

    COMMENT: "//" /[^\n]*/
    %import common.CNAME
    %import common.ESCAPED_STRING
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

    # composite-expression nodes
    def leaf(self, name):
        return canonical.compound(name)

    def and_c(self, a, b):
        return canonical.compound("and", [a, b])

    def or_c(self, a, b):
        return canonical.compound("or", [a, b])

    def not_c(self, a):
        return canonical.compound("not", [a])

    # rule-body nodes
    def atom(self, name):
        return canonical.compound(name)

    def and_b(self, left, right):
        return canonical.and_node(left, right)

    def or_b(self, left, right):
        return canonical.or_node(left, right)

    def not_b(self, inner):
        return canonical.not_node(inner)

    # declarations / rules
    def pred_decl(self, name, col, thr, cmp):
        comp = canonical.compound("predicate", [
            canonical.compound(name), col, thr, canonical.compound(cmp),
        ])
        return {"_kind": "fact", "compound": comp}

    def comp_decl(self, name, expr):
        comp = canonical.compound("composite", [canonical.compound(name), expr])
        return {"_kind": "fact", "compound": comp}

    def pos_rhs(self, name):
        return canonical.compound(str(name))

    def neg_rhs(self, name):
        # A negative goal: the rule concludes the negated target class.
        return canonical.not_node(canonical.compound(str(name)))

    def rule_def(self, _name, lhs, rhs):
        return {"_kind": "rule", "head": rhs, "body": lhs}

    def start(self, *clauses):
        return list(clauses)


_PARSER = Lark(_GRAMMAR, parser="lalr", transformer=_Tree())


def import_drools_source(source):
    return canonical.clauses_to_canonical(_PARSER.parse(source))


def import_drools_file(path):
    with open(path) as f:
        return import_drools_source(f.read())
