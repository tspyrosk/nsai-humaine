"""
End-to-end test: BCW simple scenario (breast cancer, automated training, explainability).

Scenario (from cheatsheet simple version):
  Predicates: high_clump_thickness (Clump Thickness > 3.0), low_mitoses (Mitoses < 1.5)
  Rule: "if the clump thickness is high but mitoses is low then the tumor is malignant"
  Training: automated (no Jupyter)
  Explainability: SHAP plot + rules triggered + summary

Prerequisites:
  - App running at http://localhost:8888  (docker-compose up)
  - input/breast_cancer.csv exists
  - secrets.env exists at project root
  - Run: pytest tests/test_bcw_e2e.py --headed  (drop --headed for CI)
"""

import pytest
from pathlib import Path
from playwright.sync_api import Page, expect

APP_URL = "http://localhost:8888"
PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = str(PROJECT_ROOT / "input" / "breast_cancer.csv")
ENV_PATH = str(PROJECT_ROOT / "secrets.env")

# Training can be slow — give it up to 10 minutes
TRAIN_TIMEOUT = 600_000

# Hardcoded dataset description (derived from breast_cancer.csv columns and domain knowledge)
DATASET_DOMAIN = "Medical breast cancer biopsy data from fine needle aspirate (FNA) analysis"
ROW_DESCRIPTION = "A biopsy record from a patient with measurements of cell characteristics such as clump thickness, cell uniformity, and mitoses rate"
PREDICTION_TARGET = "Whether the tumor is malignant (1) or benign (0)"
CLASS_DESCRIPTIONS = "Class 0 = Benign tumor, Class 1 = Malignant tumor"


# ── helpers ──────────────────────────────────────────────────────────────────

def wait_for_streamlit(page: Page, timeout: int = 15_000) -> None:
    """Wait for any in-progress Streamlit rerun to complete.

    Streamlit shows a 'Stop' button while running over WebSocket.
    wait_for_load_state('networkidle') does NOT catch this.
    """
    stop_btn = page.get_by_role("button", name="Stop")
    try:
        if stop_btn.is_visible():
            stop_btn.wait_for(state="hidden", timeout=timeout)
    except Exception:
        pass


def click_tab(page: Page, label: str) -> None:
    page.get_by_role("tab", name=label).click()
    page.wait_for_timeout(500)


def select_streamlit_option(page: Page, nth: int, option_text: str) -> None:
    """Open the nth Streamlit selectbox (DOM order across all tabs) and pick an option."""
    # Wait for any active rerun — a pending rerun would close the dropdown mid-interaction
    wait_for_streamlit(page)
    combobox = page.locator("[data-testid='stSelectbox']").nth(nth).locator("[role='combobox']")
    combobox.click()
    # Type to filter the list — ensures the target option is visible without scrolling
    # and scopes the match to the open popover only
    combobox.type(option_text)
    page.locator("[data-baseweb='popover'] [role='option']").filter(has_text=option_text).first.click()
    # Wait for the rerun triggered by the selection to start, then finish.
    # Using wait_for(visible) first is more reliable than a fixed sleep because
    # the Stop button may appear up to ~500ms after the click.
    stop_btn = page.get_by_role("button", name="Stop")
    try:
        stop_btn.wait_for(state="visible", timeout=2_000)
        stop_btn.wait_for(state="hidden", timeout=15_000)
    except Exception:
        pass


def multiselect_add(page: Page, container_selector: str, option_text: str) -> None:
    """Type into a Streamlit multiselect and pick the matching option."""
    container = page.locator(container_selector)
    container.locator("input").fill(option_text)
    page.locator("[role='option']").filter(has_text=option_text).first.click()
    # Explicitly close the dropdown — leaving it open causes the next widget click
    # to fire a Streamlit rerun mid-interaction (closing open dropdown = widget change).
    page.keyboard.press("Escape")
    wait_for_streamlit(page)


# ── test ─────────────────────────────────────────────────────────────────────

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


class TestBCWSimple:
    """BCW simple scenario: Env → Load → Description → Configure → Predicates → Train → Explain."""

    # ── Sidebar: Upload .env ──────────────────────────────────────────────────

    def test_00_upload_env(self, page_with_app: Page):
        page = page_with_app

        # The sidebar .env uploader is always visible
        env_input = page.locator("[data-testid='stFileUploaderDropzoneInput']").first
        env_input.set_input_files(ENV_PATH)

        # App shows ".env loaded" in the sidebar after processing
        expect(page.get_by_text(".env loaded")).to_be_visible(timeout=10_000)
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_00_upload_env.png")

    # ── Tab 1: Load ───────────────────────────────────────────────────────────

    def test_01_load_csv(self, page_with_app: Page):
        page = page_with_app
        click_tab(page, "Load")

        # "Upload CSV" radio is selected by default; pick the CSV file uploader
        # (the sidebar uploader is index 0, the main area uploader is index 1)
        file_input = page.locator("[data-testid='stFileUploaderDropzoneInput']").nth(1)
        file_input.set_input_files(CSV_PATH)

        # Wait for the dataset preview to appear
        expect(page.get_by_text("Dataset Preview")).to_be_visible(timeout=15_000)
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_01_load_csv.png")

    def test_02_fill_dataset_description(self, page_with_app: Page):
        page = page_with_app

        # Fill the four dataset description text areas
        text_areas = page.locator("[data-testid='stTextArea'] textarea")

        text_areas.nth(0).fill(DATASET_DOMAIN)       # What is this dataset about?
        text_areas.nth(1).fill(ROW_DESCRIPTION)       # What does each row represent?
        text_areas.nth(2).fill(PREDICTION_TARGET)     # What are you trying to predict?
        text_areas.nth(3).fill(CLASS_DESCRIPTIONS)    # Describe the classes (optional)

        page.get_by_role("button", name="Save Dataset Description").click()
        expect(page.get_by_text("Dataset description saved!")).to_be_visible(timeout=8_000)
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_02_dataset_description.png")

    def test_03_configure_dataset(self, page_with_app: Page):
        page = page_with_app

        # Drop "Sample code number"
        multiselect_add(page, "[data-testid='stMultiSelect']", "Sample code number")

        # Set target column to "Malignant"
        # nth(0) = Load tab's "Select target column" (first stSelectbox in DOM)
        select_streamlit_option(page, 0, "Malignant")

        # Apply
        page.get_by_role("button", name="Apply").click()
        expect(page.get_by_text("Train and test sets loaded")).to_be_visible(timeout=10_000)
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_03_configure_dataset.png")

    # ── Tab 2: Predicates ─────────────────────────────────────────────────────

    def test_04_add_predicate_high_clump_thickness(self, page_with_app: Page):
        page = page_with_app
        click_tab(page, "Predicates")

        # Select "Clump Thickness" as the feature column (outside the form)
        # nth(1) = Predicates tab's "Select Feature Column" (second stSelectbox in DOM)
        select_streamlit_option(page, 1, "Clump Thickness")

        # Fill the simple predicate form
        form = page.locator("[data-testid='stForm']").first

        form.locator("input[type='text']").fill("high_clump_thickness")

        # Threshold
        form.locator("[data-testid='stNumberInputField']").fill("3.0")

        # Comparison: "Greater"
        form.locator("[role='combobox']").click()
        page.locator("[role='option']").filter(has_text="Greater").first.click()

        form.get_by_role("button", name="Add Predicate").click()

        # Predicate should appear in the list
        expect(page.locator("strong").filter(has_text="high_clump_thickness")).to_be_visible(timeout=8_000)
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_04_predicate_clump.png")

    def test_05_add_predicate_low_mitoses(self, page_with_app: Page):
        page = page_with_app

        # Select "Mitoses" as the feature column (outside the form)
        # nth(1) = Predicates tab's "Select Feature Column" (still index 1 after first predicate added)
        select_streamlit_option(page, 1, "Mitoses")

        form = page.locator("[data-testid='stForm']").first

        form.locator("input[type='text']").fill("low_mitoses")
        form.locator("[data-testid='stNumberInputField']").fill("1.5")

        form.locator("[role='combobox']").click()
        page.locator("[role='option']").filter(has_text="Less").first.click()

        form.get_by_role("button", name="Add Predicate").click()

        expect(page.locator("strong").filter(has_text="low_mitoses")).to_be_visible(timeout=8_000)
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_05_predicate_mitoses.png")

    # ── Tab 3: Train ──────────────────────────────────────────────────────────

    def test_06_add_and_save_rule(self, page_with_app: Page):
        page = page_with_app
        click_tab(page, "Train")

        rule_text = (
            "if the clump thickness is high but mitoses is low "
            "then the tumor is malignant"
        )

        # Step 1: type the rule and send it to the LLM pipeline via "Add Rule"
        page.get_by_label("Enter rule:").fill(rule_text)
        page.get_by_role("button", name="Add Rule").click()

        # Wait for rerun so the pill appears
        stop_btn = page.get_by_role("button", name="Stop")
        try:
            stop_btn.wait_for(state="visible", timeout=2_000)
            stop_btn.wait_for(state="hidden", timeout=15_000)
        except Exception:
            pass

        # Step 2: select the pill — Streamlit renders st.pills as data-testid='stButtonGroup'
        pills_container = page.locator("[data-testid='stButtonGroup']")
        expect(pills_container).to_be_visible(timeout=10_000)
        pill = pills_container.locator("[data-testid='stBaseButton-pills']").filter(has_text="clump thickness")
        expect(pill).to_be_visible(timeout=10_000)
        pill.click()

        # Step 3: save — triggers LLM parse, can take several minutes
        page.get_by_role("button", name="Save Rules").click()

        # Step 4: wait for the generated LTN code snippet to appear (3-min timeout)
        expect(page.locator("[data-testid='stCode']")).to_be_visible(timeout=180_000)
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_06_rules_saved.png")

    def test_07_train_models(self, page_with_app: Page):
        page = page_with_app

        page.get_by_role("button", name="Train Models").click()

        # Wait for the spinner to disappear and training to finish
        # The app calls st.rerun() after training, so we wait for "Evaluate Models" to appear
        expect(page.get_by_role("button", name="Evaluate Models")).to_be_visible(
            timeout=TRAIN_TIMEOUT
        )
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_07_training_done.png")

    def test_08_evaluate_models(self, page_with_app: Page):
        page = page_with_app

        page.get_by_role("button", name="Evaluate Models").click()

        # Accuracy expander starts expanded; wait for all three model metric labels
        expect(page.get_by_text("Accuracy")).to_be_visible(timeout=20_000)
        expect(page.get_by_text("RULES", exact=True).first).to_be_visible(timeout=10_000)
        expect(page.get_by_text("MLP", exact=True).first).to_be_visible(timeout=10_000)
        expect(page.get_by_text("LTN", exact=True).first).to_be_visible(timeout=10_000)

        # Verify accuracy thresholds — values are percentage strings e.g. "92.34"
        for label, threshold in [("RULES", 10), ("MLP", 50), ("LTN", 50)]:
            value = float(
                page.locator("[data-testid='stMetric']").filter(has_text=label).first
                    .locator("[data-testid='stMetricValue']").inner_text()
            )
            assert value > threshold, f"{label} accuracy {value:.2f}% should be > {threshold}%"

        expect(page.get_by_text("Models ready!")).to_be_visible(timeout=10_000)
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_08_evaluation.png")

    # ── Tab 4: Explain ────────────────────────────────────────────────────────

    def test_09_predict_and_check_explainability(self, page_with_app: Page):
        page = page_with_app
        click_tab(page, "Explain")

        # Use sample 0 (default)
        page.get_by_role("button", name="Predict!").click()

        # Summary is the last thing rendered (after SHAP + RAG LLM call) — its presence
        # confirms the full prediction pipeline completed successfully
        expect(page.get_by_text("Summary:")).to_be_visible(timeout=120_000)
        page.screenshot(path="/home/spyros/dev/repos/nsai-humaine/tests/snapshots/snap_09_explain.png")
