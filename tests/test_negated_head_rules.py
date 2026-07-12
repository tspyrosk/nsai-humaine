"""Spec for negative goals — rules whose conclusion is the negated target class.

A rule like Prolog ``\\+ target(X) :- Body`` (or the equivalent in Datalog /
CLIPS / Drools) lowers to a ``then_part`` of ``{"operator": "NOT", "name": ...}``
so the LTN layer can constrain ``Not(target(x, f(x)))``. Every text format that
can express a negated consequent must round-trip it; SWRL cannot, so it skips.
"""

import pytest

from rules_parsing import (
    prolog_importer,
    datalog_importer,
    clips_importer,
    drools_importer,
    rule_exporter,
)


# A rule set whose single rule concludes a negative goal: (a AND NOT b) -> NOT target.
NEGATED = {
    "predicates": [
        {"name": "a", "column_index": 0, "threshold": 0.5, "comparison": "Greater", "is_boolean": False},
        {"name": "b", "column_index": 1, "threshold": 0.5, "comparison": "Less", "is_boolean": False},
    ],
    "composite_predicates": [],
    "rules": [
        {
            "if_part": {"operator": "AND", "arg1": {"name": "a"}, "arg2": {"operator": "NOT", "name": "b"}},
            "then_part": {"operator": "NOT", "name": "target"},
        }
    ],
}


@pytest.mark.parametrize("fmt, import_source", [
    ("pl", prolog_importer.import_prolog_source),
    ("dl", datalog_importer.import_datalog_source),
    ("clp", clips_importer.import_clips_source),
    ("drl", drools_importer.import_drools_source),
])
def test_negated_head_round_trips(fmt, import_source):
    text = rule_exporter.export_rule_set(NEGATED, fmt)
    assert import_source(text) == NEGATED


def test_swrl_skips_negated_head():
    text = rule_exporter.export_rule_set(NEGATED, "swrl")
    assert "# skipped" in text
    assert "-> target" not in text  # the negated consequent must not be emitted positively


def test_prolog_negated_head_lowers_to_not_operator():
    parsed = prolog_importer.import_prolog_source(
        "predicate(a, 0, 0.5, greater).\n\\+ target(X) :- a(X)."
    )
    assert parsed["rules"][0]["then_part"] == {"operator": "NOT", "name": "target"}


def test_prolog_negated_fact_is_rejected():
    # A bodiless negated clause is not a valid fact declaration.
    with pytest.raises(Exception):
        prolog_importer.import_prolog_source("\\+ predicate(a, 0, 0.5, greater).")
