"""TDD spec for the Prolog (.pl) importer.

The importer reads a Prolog file that follows the NSAI export conventions
documented at the top of ``tests/fixtures/example_rules.pl`` and returns a
Python dict that mirrors the shape the Streamlit UI keeps in session_state
(predicates, composite_predicates) plus parsed rules conforming to the
``Rule`` pydantic schema declared in ``rules_parsing/text2rules-v2.py``.
"""

import json
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def expected_json():
    with open(FIXTURES / "example_rules.json") as f:
        return json.load(f)


@pytest.fixture
def prolog_path():
    return FIXTURES / "example_rules.pl"


def test_pl_file_converts_to_expected_json(prolog_path, expected_json):
    from rules_parsing.prolog_importer import import_prolog_file

    result = import_prolog_file(str(prolog_path))

    assert result == expected_json
