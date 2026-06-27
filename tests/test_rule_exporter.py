"""Spec for rule_exporter — the inverse of the format importers.

Exporting the canonical dict and re-importing it must round-trip back to the
original dict. The text formats are checked against their importer; SWRL is
conjunction-only so it round-trips the conjunctive fixture and skips OR/NOT
rules with a comment.
"""

import json
from pathlib import Path

import pytest

from rules_parsing import rule_exporter


FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


@pytest.fixture
def full():
    return _load("example_rules.json")


@pytest.fixture
def conjunctive():
    return _load("example_rules_swrl.json")


@pytest.mark.parametrize("fmt, importer_path", [
    ("pl", "rules_parsing.prolog_importer:import_prolog_source"),
    ("dl", "rules_parsing.datalog_importer:import_datalog_source"),
    ("clp", "rules_parsing.clips_importer:import_clips_source"),
    ("drl", "rules_parsing.drools_importer:import_drools_source"),
])
def test_full_round_trips_through_importer(fmt, importer_path, full):
    module_name, func_name = importer_path.split(":")
    import_source = getattr(__import__(module_name, fromlist=[func_name]), func_name)

    text = rule_exporter.export_rule_set(full, fmt)

    assert import_source(text) == full


def test_swrl_round_trips_conjunctive(conjunctive):
    from rules_parsing.swrl_importer import import_swrl_source

    text = rule_exporter.export_rule_set(conjunctive, "swrl")

    assert import_swrl_source(text) == conjunctive


def test_swrl_skips_non_conjunctive_rule_without_error(full):
    # full has a rule with OR + NOT, which SWRL cannot express.
    text = rule_exporter.export_rule_set(full, "swrl")

    assert "# skipped non-conjunctive rule" in text
    # The conjunctive rule is still exported.
    assert "clump_thickness(?x) ^ mitoses(?x) -> target(?x)" in text


def test_json_export_is_identity(full):
    assert json.loads(rule_exporter.export_rule_set(full, "json")) == full


def test_unsupported_format_raises(full):
    with pytest.raises(ValueError):
        rule_exporter.export_rule_set(full, "xml")


def test_supported_formats_default_is_prolog():
    # The UI relies on insertion order: Prolog (.pl) must be first (the default).
    assert next(iter(rule_exporter.SUPPORTED_FORMATS)) == "pl"
