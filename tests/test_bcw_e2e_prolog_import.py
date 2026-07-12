"""
End-to-end test: BCW simple scenario via Prolog import.

Mirrors tests/test_bcw_e2e.py but replaces the manual predicate forms and
rule authoring (+ LLM parsing step) with a single Prolog (.pl) file import.

The imported file (tests/fixtures/breast_cancer_simple.pl) defines the same
two predicates and rule that the original test types in by hand:

    high_clump_thickness : Clump Thickness > 3.0
    low_mitoses          : Mitoses < 1.5
    rule                 : high_clump_thickness AND low_mitoses → malignant

Prerequisites: identical to test_bcw_e2e.py (app on :8888, input CSV, secrets.env).
"""

import pytest
from pathlib import Path
from playwright.sync_api import Page, expect

from test_bcw_e2e import (
    APP_URL,
    CSV_PATH,
    ENV_PATH,
    TRAIN_TIMEOUT,
    DATASET_DOMAIN,
    ROW_DESCRIPTION,
    PREDICTION_TARGET,
    CLASS_DESCRIPTIONS,
    wait_for_streamlit,
    click_tab,
    select_streamlit_option,
    multiselect_add,
)


PROJECT_ROOT = Path(__file__).parent.parent
PROLOG_PATH = str(PROJECT_ROOT / "tests" / "fixtures" / "breast_cancer_simple.pl")
SNAPSHOT_DIR = PROJECT_ROOT / "tests" / "snapshots"

EXPECTED_RULE_TEXT = "IF (high_clump_thickness AND low_mitoses) THEN malignant"


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def clean_model_files():
    """Delete model files before the suite so training is always forced."""
    import subprocess
    subprocess.run(
        ["docker", "exec", "nsai-humaine_nsl-app_1",
         "sh", "-c", "rm -f /app/output/ltn.h5 /app/output/mlp.h5"],
        check=False
    )
    yield


@pytest.fixture(scope="module")
def page_with_app(browser):
    """Single browser page for the whole test module (state is preserved across steps)."""
    context = browser.new_context()
    page = context.new_page()
    page.goto(APP_URL, wait_until="networkidle")
    yield page
    context.close()


# ── test ─────────────────────────────────────────────────────────────────────

class TestBCWSimpleViaProlog:
    """BCW simple scenario, Prolog-imported: Env → Load → Description → Configure → Import .pl → Train → Explain."""

    # ── Sidebar: Upload .env ──────────────────────────────────────────────────

    def test_00_upload_env(self, page_with_app: Page):
        page = page_with_app
        env_input = page.locator("[data-testid='stFileUploaderDropzoneInput']").first
        env_input.set_input_files(ENV_PATH)
        expect(page.get_by_text(".env loaded")).to_be_visible(timeout=10_000)
        page.screenshot(path=str(SNAPSHOT_DIR / "snap_pl_00_upload_env.png"))

    # ── Tab 1: Load ───────────────────────────────────────────────────────────

    def test_01_load_csv(self, page_with_app: Page):
        page = page_with_app
        click_tab(page, "Load")
        file_input = page.locator("[data-testid='stFileUploaderDropzoneInput']").nth(1)
        file_input.set_input_files(CSV_PATH)
        expect(page.get_by_text("Dataset Preview")).to_be_visible(timeout=15_000)
        page.screenshot(path=str(SNAPSHOT_DIR / "snap_pl_01_load_csv.png"))

    def test_02_fill_dataset_description(self, page_with_app: Page):
        page = page_with_app
        text_areas = page.locator("[data-testid='stTextArea'] textarea")
        text_areas.nth(0).fill(DATASET_DOMAIN)
        text_areas.nth(1).fill(ROW_DESCRIPTION)
        text_areas.nth(2).fill(PREDICTION_TARGET)
        text_areas.nth(3).fill(CLASS_DESCRIPTIONS)
        page.get_by_role("button", name="Save Dataset Description").click()
        expect(page.get_by_text("Dataset description saved!")).to_be_visible(timeout=8_000)
        page.screenshot(path=str(SNAPSHOT_DIR / "snap_pl_02_dataset_description.png"))

    def test_03_configure_dataset(self, page_with_app: Page):
        page = page_with_app
        multiselect_add(page, "[data-testid='stMultiSelect']", "Sample code number")
        select_streamlit_option(page, 0, "Malignant")
        page.get_by_role("button", name="Apply").click()
        expect(page.get_by_text("Train and test sets loaded")).to_be_visible(timeout=10_000)
        page.screenshot(path=str(SNAPSHOT_DIR / "snap_pl_03_configure_dataset.png"))

    # ── Tab 2: Predicates — import from Prolog ────────────────────────────────

    def test_04_import_prolog_file(self, page_with_app: Page):
        page = page_with_app
        click_tab(page, "Predicates")

        # Open the importer expander.
        page.get_by_text("Import predicates & rules from a file").first.click()
        page.wait_for_timeout(300)

        # Upload the .pl file. Uploaders in DOM order:
        #   nth(0) sidebar .env, nth(1) Tab 1 CSV, nth(2) Tab 2 .pl
        pl_input = page.locator("[data-testid='stFileUploaderDropzoneInput']").nth(2)
        pl_input.set_input_files(PROLOG_PATH)
        wait_for_streamlit(page)

        # Click Import (button appears once a file is staged).
        page.get_by_role("button", name="Import", exact=True).click()

        expect(page.get_by_text("Imported 2 predicates")).to_be_visible(timeout=10_000)
        page.screenshot(path=str(SNAPSHOT_DIR / "snap_pl_04_prolog_imported.png"))

    # ── Tab 2: predicates & rules populated by the import ────────────────────

    def test_05_verify_rules_loaded(self, page_with_app: Page):
        page = page_with_app

        # The imported predicates appear in the always-visible list
        # (each with a Remove button).
        expect(page.locator("strong").filter(has_text="high_clump_thickness")).to_be_visible(timeout=10_000)
        expect(page.locator("strong").filter(has_text="low_mitoses")).to_be_visible(timeout=10_000)

        # The imported rule appears in the rules section below with its encoded
        # form (the text shows both as the rule title and in the caption).
        expect(page.get_by_text(EXPECTED_RULE_TEXT).first).to_be_visible(timeout=10_000)
        page.screenshot(path=str(SNAPSHOT_DIR / "snap_pl_05_rules_loaded.png"))

    def test_06_train_models(self, page_with_app: Page):
        page = page_with_app
        click_tab(page, "Train")

        # The importer sets rules_saved=True, so the training step must be
        # visible without any further interaction (no Save Rules click needed).
        expect(page.get_by_text("Step 1: Choose Training Method")).to_be_visible(timeout=10_000)

        page.get_by_role("button", name="Train Models").click()
        expect(page.get_by_role("button", name="Evaluate Models")).to_be_visible(
            timeout=TRAIN_TIMEOUT
        )
        page.screenshot(path=str(SNAPSHOT_DIR / "snap_pl_06_training_done.png"))

    def test_07_evaluate_models(self, page_with_app: Page):
        page = page_with_app
        page.get_by_role("button", name="Evaluate Models").click()

        expect(page.get_by_text("Accuracy")).to_be_visible(timeout=20_000)
        expect(page.get_by_text("RULES", exact=True).first).to_be_visible(timeout=10_000)
        expect(page.get_by_text("MLP", exact=True).first).to_be_visible(timeout=10_000)
        expect(page.get_by_text("LTN", exact=True).first).to_be_visible(timeout=10_000)

        for label, threshold in [("RULES", 10), ("MLP", 50), ("LTN", 50)]:
            value = float(
                page.locator("[data-testid='stMetric']").filter(has_text=label).first
                    .locator("[data-testid='stMetricValue']").inner_text()
            )
            assert value > threshold, f"{label} accuracy {value:.2f}% should be > {threshold}%"

        expect(page.get_by_text("Models ready!")).to_be_visible(timeout=10_000)
        page.screenshot(path=str(SNAPSHOT_DIR / "snap_pl_07_evaluation.png"))

    # ── Tab 4: Explain ────────────────────────────────────────────────────────

    def test_08_predict_and_check_explainability(self, page_with_app: Page):
        page = page_with_app
        click_tab(page, "Explain")
        page.get_by_role("button", name="Predict!").click()
        expect(page.get_by_text("Summary:")).to_be_visible(timeout=120_000)
        page.screenshot(path=str(SNAPSHOT_DIR / "snap_pl_08_explain.png"))
