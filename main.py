import os
import time
import uuid
import zipfile
import tempfile
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
import shap
import absl.logging
import dataset.minio_utils as minio_utils
from utils import remove_rules_and_predicates, remove_outputs, remove_rules, collect_predicate_names
from dotenv import dotenv_values
import io
from paths import *

# Import business logic services
from services import data_service, model_service, explanation_service, predicate_service, image_service, text_service, notebook_service, logging_service
from services.image_service import QuotaExhaustedError as ImageQuotaError
from services.text_service import QuotaExhaustedError as TextQuotaError
from rules_parsing import rule_exporter, canonical
from PIL import Image

absl.logging.set_verbosity(absl.logging.ERROR)

st.set_page_config(layout="wide")

GLOBAL_SEED = st.sidebar.number_input("Enter a Seed for the current run:", 0, 10000, 99, placeholder="Run Seed")

SAMPLE_RATIO = st.sidebar.number_input("Data Ratio: ", value=1.0, min_value=0.1, max_value=1.0, step=0.1)
TRAIN_EPOCHS = st.sidebar.number_input("Training Epochs:", 1, 2000, 40)

env_file = st.sidebar.file_uploader("Upload .env file", type=["env"])

if env_file:
    # Only process env file once per upload (avoid re-authenticating on every Streamlit rerun)
    env_file_id = id(env_file.getvalue())
    if 'env_file_id' not in st.session_state or st.session_state.env_file_id != env_file_id:
        env_text = env_file.getvalue().decode("utf-8")
        env_vars = dotenv_values(stream=io.StringIO(env_text))
        for key, value in env_vars.items():
            os.environ[key] = value
        try:
            st.session_state.minio_token = minio_utils.minio_auth(os.getenv("MINIO_USER"), os.getenv("MINIO_PASS"))
        except Exception as e:
            st.session_state.minio_token = None
            st.sidebar.warning(f"MinIO authentication failed: {e}")
        st.session_state.env_file_id = env_file_id
    st.sidebar.success(".env loaded")
    if st.session_state.get("minio_token"):
        st.sidebar.success("MinIO authenticated")

def init_predicates_state():
    if 'predicates' not in st.session_state:
        st.session_state.predicates = []
    if 'composite_predicates' not in st.session_state:
        st.session_state.composite_predicates = []
    if 'predicates_saved' not in st.session_state:
        st.session_state.predicates_saved = False
    # Each rule is {"text": <natural-language string>, "structured": <canonical rule dict>}
    if 'rules' not in st.session_state:
        st.session_state.rules = []
    # (kind, index) of the predicate being edited inline, or None
    if 'editing_pred' not in st.session_state:
        st.session_state.editing_pred = None
    # index of the rule being edited inline, or None
    if 'editing_rule' not in st.session_state:
        st.session_state.editing_rule = None

def reset_predicates_state():
    st.session_state.predicates = []
    st.session_state.composite_predicates = []
    st.session_state.predicates_saved = False
    st.session_state.editing_pred = None
    st.session_state.editing_rule = None

def init_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'image_metadata' not in st.session_state:
        st.session_state.image_metadata = None
    if 'dataset_type' not in st.session_state:
        st.session_state.dataset_type = None
    if 'extracted_dataset_path' not in st.session_state:
        st.session_state.extracted_dataset_path = None
    if 'text_metadata' not in st.session_state:
        st.session_state.text_metadata = None
    if 'text_file_path' not in st.session_state:
        st.session_state.text_file_path = None
    if 'dataset_description' not in st.session_state:
        st.session_state.dataset_description = None
    if 'test_indices' not in st.session_state:
        st.session_state.test_indices = None
    if 'original_image_paths' not in st.session_state:
        st.session_state.original_image_paths = None
    if 'original_texts' not in st.session_state:
        st.session_state.original_texts = None
    if 'minio_token' not in st.session_state:
        st.session_state.minio_token = None
    if 'rules_saved' not in st.session_state:
        st.session_state.rules_saved = False
    if 'log_events' not in st.session_state:
        st.session_state.log_events = []
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = time.time()
    if 'run_id' not in st.session_state:
        session_ts = time.strftime(
            "%Y%m%d-%H%M%S",
            time.localtime(st.session_state.session_start_time)
        )
        st.session_state.run_id = f"{session_ts}_{uuid.uuid4().hex[:6]}"
    init_predicates_state()
    st.cache_data.clear()

def reset_state():
    st.session_state.df = None
    st.session_state.target_column = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.processed_df = None
    st.session_state.image_metadata = None
    st.session_state.dataset_type = None
    # Clean up extracted dataset directory if it exists
    if st.session_state.get('extracted_dataset_path') and os.path.exists(st.session_state.extracted_dataset_path):
        shutil.rmtree(st.session_state.extracted_dataset_path, ignore_errors=True)
    st.session_state.extracted_dataset_path = None
    st.session_state.text_metadata = None
    st.session_state.text_file_path = None
    st.session_state.dataset_description = None
    st.session_state.test_indices = None
    st.session_state.original_image_paths = None
    st.session_state.original_texts = None
    st.session_state.rules = []
    st.session_state.rules_saved = False
    reset_predicates_state()
    st.cache_data.clear()

if st.sidebar.button("Reset Rules"):
    remove_rules()
    st.session_state.rules = []
    st.session_state.rules_saved = False
    logging_service.append_event(logging_service.make_event(
        "DomainExpert", "human", "reset all rules",
        correct=False, event_type="rule_reset"
    ))
if st.sidebar.button("Reset Rules and Predicates"):
    remove_rules_and_predicates()
    st.session_state.rules = []
    st.session_state.rules_saved = False
    reset_predicates_state()
    logging_service.append_event(logging_service.make_event(
        "DomainExpert", "human", "reset all rules",
        correct=False, event_type="rule_reset"
    ))
    logging_service.append_event(logging_service.make_event(
        "DomainExpert", "human", "reset all predicates",
        correct=False, event_type="predicate_reset"
    ))
if st.sidebar.button("Reset All"):
    remove_outputs()
    reset_state()
    st.rerun()

MINIO_BUCKET = "smart-healthcare-diabetes-models"
LTN_LOCAL_PATH = os.path.join(OUTPUT_DIR, "ltn.h5")
SCALER_LOCAL_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")


def _new_version():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# Text and image models are published under a type-specific subdirectory so
# downstream consumers can distinguish them. Tabular stays at the version root
# for backwards compatibility.
_DATA_SUBDIR = {"text": "text", "image": "images"}


def _data_subdir(dataset_type):
    return _DATA_SUBDIR.get(dataset_type)


def _staging_prefix(version, data_subdir):
    return f"published-models/{version}/{data_subdir}" if data_subdir else f"published-models/{version}"


def _production_prefix(data_subdir):
    return f"production/{data_subdir}" if data_subdir else "production"


def check_and_promote(token, version):
    """Check if an approval file exists for the given version and promote to production if so.
    Returns True if promotion was performed."""
    approval = minio_utils.minio_read_json(token, MINIO_BUCKET, f"approvals/{version}/approval.json")
    if approval is None:
        return False

    # Read the version manifest to discover which type-specific subdirectory
    # this model lives under so we mirror the same layout in production/.
    version_manifest = minio_utils.minio_read_json(
        token, MINIO_BUCKET, f"published-models/{version}/manifest.json"
    ) or {}
    data_subdir = version_manifest.get("data_subdir")  # "text", "images", or None
    staging_prefix = _staging_prefix(version, data_subdir)
    prod_prefix = _production_prefix(data_subdir)

    # Copy all versioned artifacts to production (download then re-upload).
    # feature_names.txt and tag_vocabulary.json are included so the inference
    # service can reconstruct the exact training feature space from production alone.
    artifacts = [
        (f"{staging_prefix}/ltn.h5",               f"{prod_prefix}/ltn.h5"),
        (f"{staging_prefix}/scaler.pkl",            f"{prod_prefix}/scaler.pkl"),
        (f"{staging_prefix}/feature_names.txt",     f"{prod_prefix}/feature_names.txt"),
        (f"{staging_prefix}/tag_vocabulary.json",   f"{prod_prefix}/tag_vocabulary.json"),
    ]

    # The published rules file (if any) travels to production alongside the model.
    rules_path = version_manifest.get("rules_path")
    if rules_path:
        artifacts.append((rules_path, f"{prod_prefix}/{os.path.basename(rules_path)}"))

    for src, dst in artifacts:
        tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(dst))
        try:
            minio_utils.minio_download(token, MINIO_BUCKET, src, tmp_path)
            minio_utils.minio_upload(token, MINIO_BUCKET, dst, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    promoted_at = datetime.now(timezone.utc).isoformat()

    # Write production manifest (one per data type so promoting a text model
    # does not overwrite the pointer to an image model and vice versa).
    minio_utils.minio_write_json(token, MINIO_BUCKET, f"{prod_prefix}/manifest.json", {
        "version": version,
        "data_type": version_manifest.get("data_type"),
        "data_subdir": data_subdir,
        "promoted_at": promoted_at,
        "model_path": f"{staging_prefix}/ltn.h5",
        "scaler_path": f"{staging_prefix}/scaler.pkl",
        "feature_names_path": f"{staging_prefix}/feature_names.txt",
        "tag_vocabulary_path": f"{staging_prefix}/tag_vocabulary.json",
        "rules_path": rules_path,
        "rules_format": version_manifest.get("rules_format"),
    })

    # Update version manifest status
    version_manifest["status"] = "production"
    minio_utils.minio_write_json(token, MINIO_BUCKET, f"published-models/{version}/manifest.json", version_manifest)

    # Update index
    index = minio_utils.minio_read_json(token, MINIO_BUCKET, "published-models/index.json") or []
    for entry in index:
        if entry["version"] == version:
            entry["status"] = "production"
    minio_utils.minio_write_json(token, MINIO_BUCKET, "published-models/index.json", index)

    return True


@st.dialog("Publish Model")
def publish_model_dialog():
    st.write(
        "The LTN model will be published to MinIO and made available to downstream consumers. "
        "Do you want to proceed?"
    )
    rules_format = st.selectbox(
        "Rules export format",
        options=list(rule_exporter.SUPPORTED_FORMATS.keys()),
        format_func=lambda ext: rule_exporter.SUPPORTED_FORMATS[ext],
        help="The defined predicates and rules are published alongside the model in this format.",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", use_container_width=True):
            token = st.session_state.get("minio_token")
            version = _new_version()
            data_type         = st.session_state.dataset_type
            data_subdir       = _data_subdir(data_type)
            staging_prefix    = _staging_prefix(version, data_subdir)
            model_object      = f"{staging_prefix}/ltn.h5"
            scaler_object     = f"{staging_prefix}/scaler.pkl"
            feat_names_object = f"{staging_prefix}/feature_names.txt"
            tag_vocab_object  = f"{staging_prefix}/tag_vocabulary.json"

            # Upload all artifacts needed to reconstruct the training feature space.
            minio_utils.minio_upload(token, MINIO_BUCKET, model_object,      LTN_LOCAL_PATH)
            minio_utils.minio_upload(token, MINIO_BUCKET, scaler_object,     SCALER_LOCAL_PATH)
            minio_utils.minio_upload(token, MINIO_BUCKET, feat_names_object, FEATURE_NAMES_PATH)
            minio_utils.minio_upload(token, MINIO_BUCKET, tag_vocab_object,  TAG_VOCABULARY_PATH)

            # Export the canonical rule set (predicates + composite predicates +
            # rules) in the chosen format and publish it next to the model so it
            # travels with the scaler/metadata. Rules are optional.
            rules_object = None
            rule_set = predicate_service.load_rule_set()
            if rule_set:
                rules_object = f"{staging_prefix}/rules.{rules_format}"
                rules_text = rule_exporter.export_rule_set(rule_set, rules_format)
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=f".{rules_format}", delete=False
                )
                tmp.write(rules_text)
                tmp.close()
                try:
                    minio_utils.minio_upload(token, MINIO_BUCKET, rules_object, tmp.name)
                finally:
                    os.unlink(tmp.name)
            else:
                st.warning("No rules defined — publishing without a rules file.")

            # Write version manifest (the version-level manifest lives one
            # level above the data-type subdir so check_and_promote can find
            # it without knowing the data type up front).
            published_at = datetime.now(timezone.utc).isoformat()
            minio_utils.minio_write_json(token, MINIO_BUCKET, f"published-models/{version}/manifest.json", {
                "version": version,
                "data_type": data_type,
                "data_subdir": data_subdir,
                "published_at": published_at,
                "run_id": st.session_state.run_id,
                "model_path": model_object,
                "scaler_path": scaler_object,
                "feature_names_path": feat_names_object,
                "tag_vocabulary_path": tag_vocab_object,
                "rules_path": rules_object,
                "rules_format": rules_format if rules_object else None,
                "bucket": MINIO_BUCKET,
                "status": "pending_approval",
            })

            # Update index
            index = minio_utils.minio_read_json(token, MINIO_BUCKET, "published-models/index.json") or []
            index.insert(0, {
                "version": version,
                "published_at": published_at,
                "status": "pending_approval",
                "data_type": data_type,
                "data_subdir": data_subdir,
                "rules_format": rules_format if rules_object else None,
            })
            minio_utils.minio_write_json(token, MINIO_BUCKET, "published-models/index.json", index)

            logging_service.append_event(logging_service.make_event(
                "DomainExpert", "human", "approved model publish to MinIO",
                correct=True, event_type="decision"
            ))
            logging_service.publish_events(token, MINIO_BUCKET, st.session_state.run_id)
            st.success(f"Model published successfully (version: {version}).")
            st.rerun()
    with col2:
        if st.button("No", use_container_width=True):
            st.rerun()

if st.sidebar.button("Publish Model"):
    if not os.path.exists(LTN_LOCAL_PATH):
        st.error("No trained LTN model found. Please train the model before publishing.")
    elif not os.path.exists(SCALER_LOCAL_PATH):
        st.error("No scaler found (output/scaler.pkl). Please train the model before publishing.")
    elif not os.path.exists(FEATURE_NAMES_PATH):
        st.error("No feature_names.txt found. Please train the model before publishing.")
    elif not os.path.exists(TAG_VOCABULARY_PATH):
        st.error("No tag_vocabulary.json found. Please load a dataset before publishing.")
    else:
        publish_model_dialog()


init_state()

# Authenticate with MinIO if credentials are in env and we don't have a token yet
if st.session_state.minio_token is None:
    _minio_user = os.getenv("MINIO_USER")
    _minio_pass = os.getenv("MINIO_PASS")
    if _minio_user and _minio_pass:
        st.session_state.minio_token = minio_utils.minio_auth(_minio_user, _minio_pass)

def decision_plot(collected_shap_values, X_test):
    """Display SHAP decision plot."""
    st_shap(shap.decision_plot(0.5, collected_shap_values, features=X_test,
                                feature_names=data_service.get_feature_names()),
            height=500, width=800)


def write_results(results, model_name, result_name, comp_model_name=None):
    subdict = results[model_name]
    for (k, v) in subdict.items():
        if k == result_name:
            out_value = f'{(v * 100):.2f}'
            if comp_model_name is None:
                diff = 0
            else:
                baseline = results[comp_model_name][k]
                if baseline > 0:
                    diff = (v - baseline) / baseline
                else:
                    diff = 1.0
            st.metric(model_name, out_value, f'{(diff * 100):+.2f}' + "%")


def create_metric_expander(results, result_name, start_expanded=False):
    with st.expander(result_name, expanded=start_expanded):
        col1, col2, col3 = st.columns(3)
        with col1:
            write_results(results, "RULES", result_name)
        with col2:
            write_results(results, "MLP", result_name, comp_model_name="RULES")
        with col3:
            if "LTN" in results:
                write_results(results, "LTN", result_name, comp_model_name="MLP")


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Load", "Predicates & Rules", "Train", "Explain", "Models"])

with tab1:
    st.header("Upload Dataset")

    upload_method = st.radio("Choose data source", ["Upload CSV", "MinIO Path", "Image Dataset", "Text Dataset"])

    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            st.session_state.df = data_service.save_uploaded_csv(
                uploaded_file.getbuffer(),
                INPUT_CSV
            )
            st.session_state.dataset_type = "csv"
            import json as _json
            with open(TAG_VOCABULARY_PATH, "w") as _f:
                _json.dump({"data_type": "tabular", "tags": [], "image_feature_names": []}, _f)

    elif upload_method == "MinIO Path":
        minio_path = st.text_input("Enter MinIO Path")
        if st.button("Load from MinIO"):
            if minio_path:
                try:
                    bucket = minio_path.split("/")[0]
                    path = "/".join(minio_path.split("/")[1:])
                    print(f"Reading from MinIO. Bucket: {bucket}, rePath: {path}")
                    minio_utils.minio_download(st.session_state.minio_token, bucket, path, INPUT_CSV)
                    st.session_state.df = pd.read_csv(INPUT_CSV)
                    st.session_state.dataset_type = "csv"
                    import json as _json
                    with open(TAG_VOCABULARY_PATH, "w") as _f:
                        _json.dump({"data_type": "tabular", "tags": [], "image_feature_names": []}, _f)
                except Exception as e:
                    st.error(f"Error loading data from MinIO: {str(e)}")
            else:
                st.warning("Please enter a valid MinIO path.")

    elif upload_method == "Image Dataset":
        st.subheader("Load Image Dataset")
        st.info("Upload a ZIP file containing two folders (one for each class). "
                "Example structure: `dataset.zip` → `class_a/` and `class_b/` folders with images inside.")

        uploaded_zip = st.file_uploader("Upload ZIP file with image dataset", type=["zip"])

        if uploaded_zip is not None:
            # Create a persistent directory for the extracted dataset
            dataset_name = os.path.splitext(uploaded_zip.name)[0]
            dataset_path = os.path.join(INPUT_DIR, f"uploaded_{dataset_name}")

            # Check if we need to extract (avoid re-extracting on every rerun).
            # Preserve an existing extracted folder across sessions so its cache
            # files (tags_cache.json, image_features_cache.json, dataset_hash.txt)
            # survive re-uploads of the same ZIP.
            if 'extracted_dataset_path' not in st.session_state or st.session_state.extracted_dataset_path != dataset_path:
                if not os.path.exists(dataset_path):
                    with st.spinner("Extracting ZIP file..."):
                        os.makedirs(dataset_path, exist_ok=True)
                        with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()), 'r') as zip_ref:
                            zip_ref.extractall(dataset_path)

                        # Handle case where ZIP contains a single root folder
                        extracted_items = [f for f in os.listdir(dataset_path) if not f.startswith('.')]
                        if len(extracted_items) == 1:
                            single_folder = os.path.join(dataset_path, extracted_items[0])
                            if os.path.isdir(single_folder):
                                for item in os.listdir(single_folder):
                                    shutil.move(os.path.join(single_folder, item), dataset_path)
                                os.rmdir(single_folder)
                    st.success(f"ZIP extracted to: {dataset_path}")
                else:
                    st.info(f"Reusing existing extracted dataset at: {dataset_path}")

                st.session_state.extracted_dataset_path = dataset_path

            # Load and display the dataset
            try:
                image_paths, labels, class_names = image_service.load_image_dataset(dataset_path)

                st.write(f"**Classes:** {', '.join(class_names)}")
                st.write(f"**Total images:** {len(image_paths)} "
                        f"(Class 0: {len(labels) - sum(labels)}, Class 1: {sum(labels)})")

                # Display sample images from each class
                st.subheader("Sample Images")
                samples = image_service.get_sample_images(image_paths, labels, n_per_class=4)

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**{class_names[0] if labels[0] == 0 else class_names[1]}** (Class 0 - Negative)")
                    sample_cols = st.columns(4)
                    for idx, img_path in enumerate(samples.get(0, [])[:4]):
                        with sample_cols[idx]:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)

                with col2:
                    positive_class_name = class_names[1] if labels[0] == 0 else class_names[0]
                    st.write(f"**{positive_class_name}** (Class 1 - Positive)")
                    sample_cols = st.columns(4)
                    for idx, img_path in enumerate(samples.get(1, [])[:4]):
                        with sample_cols[idx]:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)

                # Dataset description for tag extraction
                st.subheader("Dataset Context (Optional)")
                st.info("Providing context helps extract more relevant tags from images.")

                img_dataset_domain = st.text_input(
                    "What is this image dataset about?",
                    value="",
                    placeholder="e.g., Medical X-rays, Manufacturing defect detection, Satellite imagery",
                    key="img_domain"
                )
                img_row_description = st.text_input(
                    "What does each image represent?",
                    value="",
                    placeholder="e.g., A chest X-ray, A product photo, A satellite image of terrain",
                    key="img_row_desc"
                )
                img_prediction_target = st.text_input(
                    "What are you trying to classify?",
                    value="",
                    placeholder="e.g., Presence of pneumonia, Product defect type, Land use category",
                    key="img_pred_target"
                )

                # Build image dataset description
                img_dataset_description = None
                if img_dataset_domain or img_row_description or img_prediction_target:
                    img_dataset_description = {
                        'domain': img_dataset_domain,
                        'row_description': img_row_description,
                        'prediction_target': img_prediction_target,
                        'class_descriptions': ''
                    }

                # Option to force refresh cache
                force_refresh = st.checkbox("Force refresh (re-extract all features)", value=False)

                # Check if OpenAI API key is available
                openai_key = os.getenv("OPENAI_API_KEY")
                if not openai_key:
                    st.info("OpenAI API key not found. Tag extraction will be skipped. "
                           "Upload a .env file with OPENAI_API_KEY to enable tag extraction.")

                # Process button
                if st.button("Process Image Dataset"):
                    with st.spinner("Processing images..."):
                        # Progress indicators
                        progress_text = st.empty()
                        progress_bar = st.progress(0)

                        def update_progress(current, total, stage=""):
                            progress = current / total if total > 0 else 1.0
                            progress_bar.progress(progress)
                            progress_text.text(f"{stage}: {current}/{total}")

                        try:
                            # Extract features and create tabular dataset
                            df, metadata = image_service.load_image_dataset_as_tabular(
                                dataset_path,
                                openai_api_key=openai_key,
                                force_refresh=force_refresh,
                                tag_progress_callback=lambda c, t: update_progress(c, t, "Extracting tags"),
                                feature_progress_callback=lambda c, t: update_progress(c, t, "Extracting image features"),
                                dataset_description=img_dataset_description
                            )

                            # Store dataset description in session state
                            if img_dataset_description:
                                st.session_state.dataset_description = img_dataset_description

                            # Save as CSV for compatibility with existing data flow
                            df.to_csv(INPUT_CSV, index=False)

                            # Save tag vocabulary so the inference service can
                            # constrain its LLM prompts to the training tags.
                            import json as _json
                            with open(TAG_VOCABULARY_PATH, "w") as _f:
                                _json.dump({
                                    "data_type": "image",
                                    "tags": list(metadata["all_tags"]),
                                    "image_feature_names": list(metadata["image_feature_names"]),
                                }, _f, indent=2)

                            # Store in session state
                            st.session_state.df = df
                            st.session_state.image_metadata = metadata
                            st.session_state.dataset_type = "image"

                            progress_bar.progress(1.0)
                            progress_text.text("Processing complete!")

                            # Show success with standardization stats
                            tag_stats = metadata.get('tag_standardization', {})
                            if tag_stats and not tag_stats.get('skipped'):
                                st.success(f"Image dataset converted to tabular format! "
                                          f"({tag_stats.get('original_count', 0)} tags extracted → "
                                          f"{len(metadata['all_tags'])} after semantic merging + "
                                          f"{len(metadata['image_feature_names'])} image features)")
                            else:
                                st.success(f"Image dataset converted to tabular format! "
                                          f"({len(metadata['all_tags'])} tag features + "
                                          f"{len(metadata['image_feature_names'])} image features)")

                        except ImageQuotaError as e:
                            st.error("OpenAI API quota exhausted. Please check your billing settings at https://platform.openai.com/account/billing")
                        except Exception as e:
                            st.error(f"Error processing image dataset: {str(e)}")

            except Exception as e:
                st.error(f"Error loading image dataset: {str(e)}")

    elif upload_method == "Text Dataset":
        st.subheader("Load Text Dataset")
        st.info("Upload a CSV or TSV file containing text data with labels. "
                "Example: SMS spam dataset with 'text' and 'label' columns.")

        uploaded_text_file = st.file_uploader("Upload text dataset file (CSV, TSV, or TXT)", type=None)

        if uploaded_text_file is not None:
            # Save the uploaded file
            file_name = uploaded_text_file.name
            file_path = os.path.join(INPUT_DIR, f"uploaded_{file_name}")

            # Check if we need to save (avoid re-saving on every rerun)
            if 'text_file_path' not in st.session_state or st.session_state.text_file_path != file_path:
                with open(file_path, 'wb') as f:
                    f.write(uploaded_text_file.getvalue())
                st.session_state.text_file_path = file_path
                st.success(f"File saved to: {file_path}")

            try:
                # Load and preview the dataset
                texts, labels, label_names = text_service.load_text_dataset(file_path)

                st.write(f"**Labels:** {', '.join(label_names)}")
                st.write(f"**Total texts:** {len(texts)} "
                        f"(Class 0: {len(labels) - sum(labels)}, Class 1: {sum(labels)})")

                # Display sample texts from each class
                st.subheader("Sample Texts")
                samples = text_service.get_sample_texts(texts, labels, n_per_class=3)

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**{label_names[0]}** (Class 0 - Negative)")
                    for idx, text in enumerate(samples.get(0, [])[:3]):
                        with st.expander(f"Sample {idx + 1}", expanded=idx == 0):
                            st.write(text[:500] + "..." if len(text) > 500 else text)

                with col2:
                    st.write(f"**{label_names[1] if len(label_names) > 1 else 'Class 1'}** (Class 1 - Positive)")
                    for idx, text in enumerate(samples.get(1, [])[:3]):
                        with st.expander(f"Sample {idx + 1}", expanded=idx == 0):
                            st.write(text[:500] + "..." if len(text) > 500 else text)

                # Dataset description for tag extraction
                st.subheader("Configuration")
                st.info("Providing context helps extract more relevant tags and generate better explanations.")

                text_dataset_domain = st.text_input(
                    "What is this dataset about?",
                    value="",
                    placeholder="e.g., SMS spam detection, Email classification, Customer reviews",
                    key="text_domain"
                )
                text_row_description = st.text_input(
                    "What does each row represent?",
                    value="",
                    placeholder="e.g., An SMS message, An email, A customer review",
                    key="text_row_desc"
                )
                text_prediction_target = st.text_input(
                    "What are you trying to classify?",
                    value="",
                    placeholder="e.g., Whether the message is spam, Sentiment of the review",
                    key="text_pred_target"
                )
                text_class_descriptions = st.text_input(
                    "Describe the classes (optional):",
                    value="",
                    placeholder="e.g., ham = legitimate message, spam = unwanted message",
                    key="text_class_desc"
                )

                # Build description string for tag extraction
                dataset_description_parts = []
                if text_dataset_domain:
                    dataset_description_parts.append(text_dataset_domain)
                if text_row_description:
                    dataset_description_parts.append(f"Each row is {text_row_description}")
                if text_prediction_target:
                    dataset_description_parts.append(f"Goal: {text_prediction_target}")

                dataset_description = ". ".join(dataset_description_parts) if dataset_description_parts else "Text messages labeled for binary classification."

                # Build structured dataset description for session state
                text_dataset_description = {
                    'domain': text_dataset_domain,
                    'row_description': text_row_description,
                    'prediction_target': text_prediction_target,
                    'class_descriptions': text_class_descriptions
                }

                # Batch size configuration
                batch_size = st.number_input(
                    "Batch size for tag extraction:",
                    min_value=5, max_value=250, value=150,
                    help="Number of texts to process in each API call. Higher values are faster but may hit rate limits."
                )

                # Max tags configuration
                max_tags = st.number_input(
                    "Maximum tags to keep:",
                    min_value=10, max_value=100, value=30,
                    help="Similar tags are merged and only the top N most discriminative tags are kept."
                )

                # Option to force refresh cache
                force_refresh = st.checkbox("Force refresh (re-extract all tags)", value=False)

                # Check if OpenAI API key is available
                openai_key = os.getenv("OPENAI_API_KEY")
                if not openai_key:
                    st.warning("OpenAI API key not found. Tag extraction will be skipped. "
                             "Upload a .env file with OPENAI_API_KEY to enable tag extraction.")

                # Process button
                if st.button("Process Text Dataset"):
                    with st.spinner("Processing texts..."):
                        # Progress indicators
                        progress_text = st.empty()
                        progress_bar = st.progress(0)

                        def update_progress(current, total):
                            progress = current / total if total > 0 else 1.0
                            progress_bar.progress(progress)
                            progress_text.text(f"Extracting tags: {current}/{total} texts processed")

                        try:
                            # Extract features and create tabular dataset
                            df, metadata = text_service.load_text_dataset_as_tabular(
                                file_path,
                                dataset_description=dataset_description,
                                openai_api_key=openai_key,
                                force_refresh=force_refresh,
                                batch_size=batch_size,
                                max_tags=max_tags,
                                progress_callback=update_progress
                            )

                            # Save as CSV for compatibility with existing data flow
                            df.to_csv(INPUT_CSV, index=False)

                            # Save tag vocabulary so the inference service can
                            # constrain its LLM prompts to the training tags.
                            import json as _json
                            with open(TAG_VOCABULARY_PATH, "w") as _f:
                                _json.dump({
                                    "data_type": "text",
                                    "tags": list(metadata["all_tags"]),
                                    "image_feature_names": [],
                                }, _f, indent=2)

                            # Store in session state
                            st.session_state.df = df
                            st.session_state.text_metadata = metadata
                            st.session_state.dataset_type = "text"

                            # Store dataset description in session state
                            st.session_state.dataset_description = text_dataset_description

                            progress_bar.progress(1.0)
                            progress_text.text("Processing complete!")

                            # Show reduction stats
                            reduction_stats = metadata.get('reduction_stats', {})
                            st.success(f"Text dataset converted to tabular format! "
                                      f"({reduction_stats.get('original_tag_count', 0)} tags extracted → "
                                      f"{reduction_stats.get('after_merge_count', 0)} after merging → "
                                      f"{len(metadata['all_tags'])} final tags)")

                        except TextQuotaError as e:
                            st.error("OpenAI API quota exhausted. Please check your billing settings at https://platform.openai.com/account/billing")
                        except Exception as e:
                            st.error(f"Error processing text dataset: {str(e)}")

            except Exception as e:
                st.error(f"Error loading text dataset: {str(e)}")

    # Common section for all data sources - display DataFrame and configure
    if st.session_state.df is not None:
        st.divider()
        st.subheader("Dataset Preview")
        st.write(st.session_state.df.head())

        # Dataset description section - helps make prompts domain-agnostic
        st.divider()
        st.subheader("Dataset Description")
        st.info("Provide context about your dataset. This helps generate more relevant explanations and feature extraction.")

        col_desc1, col_desc2 = st.columns(2)
        with col_desc1:
            dataset_domain = st.text_area(
                "What is this dataset about?",
                value=st.session_state.dataset_description.get('domain', '') if st.session_state.dataset_description else '',
                placeholder="e.g., Medical diagnosis data, Manufacturing quality control, SMS spam detection, Customer churn prediction",
                help="Briefly describe the domain and purpose of this dataset."
            )
            row_description = st.text_area(
                "What does each row represent?",
                value=st.session_state.dataset_description.get('row_description', '') if st.session_state.dataset_description else '',
                placeholder="e.g., A patient's medical record, A product from the assembly line, An SMS message, A customer account",
                help="Describe what a single data point or observation represents."
            )
        with col_desc2:
            prediction_target = st.text_area(
                "What are you trying to predict?",
                value=st.session_state.dataset_description.get('prediction_target', '') if st.session_state.dataset_description else '',
                placeholder="e.g., Whether the patient has diabetes, Whether the product is defective, Whether the message is spam",
                help="Describe the classification goal."
            )
            class_descriptions = st.text_area(
                "Describe the classes (optional)",
                value=st.session_state.dataset_description.get('class_descriptions', '') if st.session_state.dataset_description else '',
                placeholder="e.g., Class 0 = Healthy, Class 1 = Diabetic",
                help="Provide meaning for each class label."
            )

        if st.button("Save Dataset Description"):
            st.session_state.dataset_description = {
                'domain': dataset_domain,
                'row_description': row_description,
                'prediction_target': prediction_target,
                'class_descriptions': class_descriptions
            }
            logging_service.append_event(logging_service.make_event(
                "DomainExpert", "human",
                f"saved dataset description: domain={dataset_domain}, target={prediction_target}",
                event_type="info"
            ))
            st.success("Dataset description saved!")

        # For image datasets, show metadata
        if st.session_state.dataset_type == "image" and st.session_state.image_metadata:
            meta = st.session_state.image_metadata
            with st.expander("Image Dataset Details"):
                st.write(f"**Tag features:** {len(meta['all_tags'])}")
                st.write(f"**Image features:** {len(meta['image_feature_names'])}")

                # Show tag standardization stats if available
                tag_stats = meta.get('tag_standardization', {})
                if tag_stats and not tag_stats.get('skipped'):
                    st.write(f"**Tag standardization:** {tag_stats.get('original_count', 0)} original → "
                            f"{tag_stats.get('canonical_count', 0)} after semantic merging "
                            f"({tag_stats.get('merged_groups', 0)} groups merged)")

                st.write(f"**Sample tags:** {', '.join(meta['all_tags'][:10])}...")

        # For text datasets, show metadata
        if st.session_state.dataset_type == "text" and st.session_state.text_metadata:
            meta = st.session_state.text_metadata
            with st.expander("Text Dataset Details"):
                st.write(f"**Tag features:** {len(meta['all_tags'])}")
                st.write(f"**Total texts:** {meta['n_texts']}")
                st.write(f"**Class distribution:** {meta['n_negative']} negative, {meta['n_positive']} positive")

                # Show reduction stats if available
                reduction_stats = meta.get('reduction_stats', {})
                if reduction_stats:
                    st.write(f"**Tag standardization:** {reduction_stats.get('original_tag_count', 0)} original → "
                            f"{reduction_stats.get('after_merge_count', 0)} after semantic merging → "
                            f"{reduction_stats.get('final_tag_count', 0)} final (top discriminative)")
                    st.write(f"**Semantically merged groups:** {reduction_stats.get('merged_groups', 0)}")

                    # Show sample of merged clusters
                    merged_sample = reduction_stats.get('merged_clusters_sample', {})
                    if merged_sample:
                        st.write("**Sample merged tags:**")
                        for canonical, merged in list(merged_sample.items())[:3]:
                            st.write(f"  • {canonical} ← {', '.join(merged)}")

                st.write(f"**Selected tags:** {', '.join(meta['all_tags'][:15])}{'...' if len(meta['all_tags']) > 15 else ''}")

        columns = st.session_state.df.columns.tolist()

        # Filter out non-feature columns for dropping
        droppable_columns = [c for c in columns if c not in ['image_path', 'text']]
        columns_to_drop = st.multiselect("Select columns to drop", droppable_columns)

        # For image datasets, default target to 'label'
        default_target_idx = columns.index('label') if 'label' in columns else 0
        st.session_state.target_column = st.selectbox("Select target column", columns, index=default_target_idx)

        if st.button("Apply"):
            # Store original data for display in Explain tab before dropping
            original_df = pd.read_csv(INPUT_CSV)
            if 'image_path' in original_df.columns:
                st.session_state.original_image_paths = original_df['image_path'].tolist()
            if 'text' in original_df.columns:
                st.session_state.original_texts = original_df['text'].tolist()

            # Always drop non-feature columns if they exist
            cols_to_drop = columns_to_drop.copy()
            if 'image_path' in columns and 'image_path' not in cols_to_drop:
                cols_to_drop.append('image_path')
            if 'text' in columns and 'text' not in cols_to_drop:
                cols_to_drop.append('text')

            X_train, X_test, y_test, processed_df, test_indices = data_service.load_data_from_csv(
                INPUT_CSV,
                st.session_state.target_column,
                cols_to_drop,
                GLOBAL_SEED,
                SAMPLE_RATIO
            )
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.processed_df = processed_df
            st.session_state.test_indices = test_indices

            st.success("Train and test sets loaded into session state!")


with tab2:
    st.header("Define Predicates & Rules")

    # ---- Import predicates & rules from a file (all supported formats) ----
    with st.expander("Import predicates & rules from a file "
                     "(Prolog, Datalog, SWRL, CLIPS, Drools, decision tree)"):
        st.caption(
            "Skip the manual setup by uploading a rule file. Supported formats: "
            "Prolog (.pl), Datalog (.dl), SWRL (.swrl), CLIPS/Jess (.clp), "
            "Drools DRL (.drl), or a pickled/joblib scikit-learn decision tree "
            "(.pkl/.joblib). The format is detected from the file extension. "
        )
        rules_upload = st.file_uploader(
            "Upload rule file",
            type=["pl", "dl", "swrl", "clp", "drl", "pkl", "joblib"],
            key="rules_format_upload",
        )
        if rules_upload is not None and st.button("Import", key="rules_format_import_btn"):
            if st.session_state.target_column is None:
                st.warning("Load a dataset and select a target column before importing.")
            else:
                ext = rules_upload.name.rsplit(".", 1)[-1].lower()
                importer = predicate_service.IMPORTERS_BY_EXT.get(ext)
                if importer is None:
                    st.error(f"Unsupported file type: .{ext}")
                else:
                    try:
                        result = importer(
                            rules_upload.getvalue(),
                            st.session_state.target_column,
                        )
                        st.session_state.predicates = result["predicates"]
                        st.session_state.composite_predicates = result["composite_predicates"]
                        st.session_state.rules = [
                            {"text": text, "structured": rule}
                            for text, rule in zip(result["rule_texts"], result["rules"])
                        ]
                        # The importer wrote the full artifact set, so both steps
                        # start out saved; any add/remove marks them dirty again.
                        st.session_state.predicates_saved = True
                        st.session_state.rules_saved = True
                        logging_service.append_event(logging_service.make_event(
                            "DomainExpert", "human",
                            f"imported {len(result['predicates'])} predicate(s), "
                            f"{len(result['composite_predicates'])} composite(s), "
                            f"{len(result['rule_texts'])} rule(s) from .{ext} file",
                            event_type="rule_authoring",
                        ))
                        st.success(
                            f"Imported {len(result['predicates'])} predicates, "
                            f"{len(result['composite_predicates'])} composite predicates, "
                            f"and {len(result['rule_texts'])} rules from .{ext} file. "
                            "Review the imported predicates and rules below."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to import .{ext} file: {e}")

    def name_exists(name):
        return predicate_service.check_predicate_name_exists(
            name,
            st.session_state.predicates,
            st.session_state.composite_predicates
        )
    
    # Helper function to detect if a column is boolean (only 0 and 1 values)
    def is_boolean_column(df, col_name):
        """Check if a column contains only boolean values (0 and 1)."""
        unique_vals = df[col_name].dropna().unique()
        return set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False})

    def describe_predicate(pred):
        col_name = pred.get('column_name', f"Column {pred['column_index']}")
        if pred.get('is_boolean', False):
            return f"{col_name} = True"
        return f"{col_name} {pred['comparison']} {pred['threshold']}"

    def get_feature_columns():
        """Return (columns, boolean_columns) for the loaded dataset.

        Boolean detection uses the ORIGINAL df (before normalization) because
        processed_df is StandardScaler-normalized, turning 0/1 into ±1.0.
        """
        columns = st.session_state.processed_df.columns.tolist()
        original_df = st.session_state.df
        if original_df is not None:
            boolean_columns = set(
                col for col in columns
                if col in original_df.columns and is_boolean_column(original_df, col)
            )
        else:
            boolean_columns = set()
        return columns, boolean_columns

    def mark_predicates_dirty():
        """Predicate changes invalidate the saved predicate files, and — because
        rules.json embeds the predicate definitions — the saved rule artifacts too."""
        st.session_state.predicates_saved = False
        if st.session_state.rules:
            st.session_state.rules_saved = False

    def start_pred_edit(kind, index):
        st.session_state.editing_pred = (kind, index)

    def start_rule_edit(index):
        st.session_state.editing_rule = index

    def remove_predicate(kind, index):
        """Button callback: remove a predicate unless something still references it."""
        collection = (st.session_state.predicates if kind == 'simple'
                      else st.session_state.composite_predicates)
        pred = collection[index]
        references = predicate_service.find_references(
            pred['name'],
            st.session_state.composite_predicates,
            st.session_state.rules,
        )
        if references:
            # Callbacks run before the script renders, so stash the warning
            # in session state for the upcoming render to display.
            st.session_state.removal_blocked = (pred['name'], references)
            return
        collection.pop(index)
        st.session_state.editing_pred = None
        mark_predicates_dirty()
        logging_service.append_event(logging_service.make_event(
            "DomainExpert", "human",
            f"removed predicate: {pred['name']}",
            event_type="predicate_removal"
        ))

    def validate_rename(old_name, new_name):
        """Return an error message if renaming old→new is not allowed, else None."""
        if new_name == old_name:
            return None
        if not new_name:
            return "Predicate name cannot be empty."
        references = predicate_service.find_references(
            old_name, st.session_state.composite_predicates, st.session_state.rules)
        if references:
            return (f"Cannot rename '{old_name}' — it is still used by "
                    f"{'; '.join(references)}. Remove those first or keep the name.")
        if name_exists(new_name):
            return f"A predicate named '{new_name}' already exists."
        return None

    def cancel_pred_edit():
        st.session_state.editing_pred = None

    def cancel_rule_edit():
        st.session_state.editing_rule = None

    def apply_simple_edit(index):
        """Button callback: apply an inline simple-predicate edit from its widget state."""
        pred = st.session_state.predicates[index]
        new_name = st.session_state[f"edit_name_{index}"].strip()
        error = validate_rename(pred['name'], new_name)
        if error:
            st.session_state.edit_error = error
            return
        columns, boolean_columns = get_feature_columns()
        new_column = st.session_state[f"edit_col_{index}"]
        if new_column in boolean_columns:
            new_threshold, new_comparison, new_is_boolean = 0.5, 'Greater', True
        else:
            new_threshold = st.session_state[f"edit_thr_{index}"]
            new_comparison = st.session_state[f"edit_cmp_{index}"]
            new_is_boolean = False
        st.session_state.predicates[index] = {
            'name': new_name,
            'column_index': columns.index(new_column),
            'threshold': new_threshold,
            'comparison': new_comparison,
            'is_boolean': new_is_boolean,
            'column_name': new_column,
        }
        st.session_state.editing_pred = None
        mark_predicates_dirty()
        logging_service.append_event(logging_service.make_event(
            "DomainExpert", "human",
            f"edited predicate: {pred['name']} -> {new_name} "
            f"({describe_predicate(st.session_state.predicates[index])})",
            event_type="predicate_edit"
        ))

    def render_simple_edit_form(index, pred):
        with st.container(border=True):
            st.markdown(f"Editing **{pred['name']}**")
            if st.session_state.get('edit_error'):
                st.warning(st.session_state.pop('edit_error'))
            if st.session_state.processed_df is None:
                st.warning("Load and apply the dataset to edit this predicate.")
                st.button("Cancel", key=f"edit_cancel_{index}", on_click=cancel_pred_edit)
                return
            columns, boolean_columns = get_feature_columns()
            current_col = pred.get('column_name')
            if current_col not in columns:
                current_col = columns[pred['column_index']] if pred['column_index'] < len(columns) else columns[0]
            new_column = st.selectbox("Feature Column", columns,
                                      index=columns.index(current_col),
                                      key=f"edit_col_{index}")
            st.text_input("Predicate Name", value=pred['name'], key=f"edit_name_{index}")
            if new_column in boolean_columns:
                st.caption(f"📌 **{new_column}** is a boolean feature (values: 0 or 1)")
            else:
                st.number_input("Threshold Value", value=float(pred['threshold']),
                                format="%f", key=f"edit_thr_{index}")
                comparisons = ["Greater", "Less", "Equal"]
                cmp_index = comparisons.index(pred['comparison']) if pred['comparison'] in comparisons else 0
                st.selectbox("Comparison Operator", comparisons,
                             index=cmp_index, key=f"edit_cmp_{index}")

            col_save, col_cancel = st.columns([1, 1])
            col_save.button("Save changes", key=f"edit_save_{index}", type="primary",
                            on_click=apply_simple_edit, args=(index,))
            col_cancel.button("Cancel", key=f"edit_cancel_{index}", on_click=cancel_pred_edit)

    def _unpack_binary_composite(expression):
        """If the expression matches the two-argument builder pattern (And/Or of
        two optionally-negated leaves), return its parts for prefilling the
        builder widgets; otherwise None (imported composites can nest deeper)."""
        try:
            node = canonical.parse_composite_expression(expression)
        except Exception:
            return None
        if node["name"] not in ("And", "Or") or len(node["args"]) != 2:
            return None

        def leaf(arg):
            if not arg["args"]:
                return arg["name"], False
            if arg["name"] == "Not" and len(arg["args"]) == 1 and not arg["args"][0]["args"]:
                return arg["args"][0]["name"], True
            return None

        left, right = leaf(node["args"][0]), leaf(node["args"][1])
        if left is None or right is None:
            return None
        return left[0], left[1], node["name"].upper(), right[0], right[1]

    def apply_composite_edit(index):
        """Button callback: apply an inline composite-predicate edit from its widget state."""
        comp = st.session_state.composite_predicates[index]
        new_name = st.session_state[f"edit_comp_name_{index}"].strip()
        error = validate_rename(comp['name'], new_name)
        new_expression = comp['expression']
        if error is None:
            if f"edit_comp_expr_{index}" in st.session_state:
                # Raw-expression fallback for composites the builder can't represent.
                new_expression = st.session_state[f"edit_comp_expr_{index}"]
                try:
                    canonical.parse_composite_expression(new_expression)
                except Exception as e:
                    error = f"Invalid expression: {e}"
            else:
                left_pred = st.session_state[f"edit_comp_left_{index}"]
                left_unary = st.session_state[f"edit_comp_lu_{index}"]
                right_pred = st.session_state[f"edit_comp_right_{index}"]
                right_unary = st.session_state[f"edit_comp_ru_{index}"]
                binary_op = st.session_state[f"edit_comp_op_{index}"]
                left_expr = f"Not({left_pred})" if left_unary == "NOT" else left_pred
                right_expr = f"Not({right_pred})" if right_unary == "NOT" else right_pred
                new_expression = f"{binary_op.capitalize()}({left_expr}, {right_expr})"
        if error:
            st.session_state.edit_error = error
            return
        st.session_state.composite_predicates[index] = {
            'name': new_name,
            'expression': new_expression,
        }
        st.session_state.editing_pred = None
        mark_predicates_dirty()
        logging_service.append_event(logging_service.make_event(
            "DomainExpert", "human",
            f"edited composite predicate: {comp['name']} -> {new_name} = {new_expression}",
            event_type="predicate_edit"
        ))

    def render_composite_edit_form(index, comp):
        with st.container(border=True):
            st.markdown(f"Editing **{comp['name']}**")
            if st.session_state.get('edit_error'):
                st.warning(st.session_state.pop('edit_error'))
            st.text_input("Composite Predicate Name", value=comp['name'],
                          key=f"edit_comp_name_{index}")

            predicate_names = [pred['name'] for pred in st.session_state.predicates]
            unpacked = _unpack_binary_composite(comp['expression'])
            builder_ok = (unpacked is not None
                          and unpacked[0] in predicate_names
                          and unpacked[3] in predicate_names)

            if builder_ok:
                left_name, left_not, op, right_name, right_not = unpacked
                unary_opts = ["No Unary Predicate", "NOT"]
                col1, col2 = st.columns(2)
                with col1:
                    st.selectbox("Left Predicate", predicate_names,
                                 index=predicate_names.index(left_name),
                                 key=f"edit_comp_left_{index}")
                    st.selectbox("Left Unary Operator", unary_opts,
                                 index=1 if left_not else 0,
                                 key=f"edit_comp_lu_{index}")
                with col2:
                    st.selectbox("Right Predicate", predicate_names,
                                 index=predicate_names.index(right_name),
                                 key=f"edit_comp_right_{index}")
                    st.selectbox("Right Unary Operator", unary_opts,
                                 index=1 if right_not else 0,
                                 key=f"edit_comp_ru_{index}")
                st.selectbox("Binary Operator", ["AND", "OR"],
                             index=0 if op == "AND" else 1,
                             key=f"edit_comp_op_{index}")
            else:
                # Imported composites can be arbitrarily nested — edit the raw expression.
                st.caption("This expression doesn't fit the two-predicate builder, so edit it "
                           "directly (And/Or/Not over predicate names, e.g. `And(a, Not(b))`).")
                st.text_input("Expression", value=comp['expression'],
                              key=f"edit_comp_expr_{index}")

            col_save, col_cancel = st.columns([1, 1])
            col_save.button("Save changes", key=f"edit_comp_save_{index}", type="primary",
                            on_click=apply_composite_edit, args=(index,))
            col_cancel.button("Cancel", key=f"edit_comp_cancel_{index}", on_click=cancel_pred_edit)


    # ---- Always-visible list of defined predicates ----
    st.subheader("Defined Predicates")

    if st.session_state.get('removal_blocked'):
        blocked_name, blocked_refs = st.session_state.pop('removal_blocked')
        st.warning(
            f"Cannot remove '{blocked_name}' — it is still used by "
            f"{'; '.join(blocked_refs)}. Remove those first."
        )

    if not st.session_state.predicates and not st.session_state.composite_predicates:
        st.caption("No predicates defined yet — add one below or import a rule file above.")
    for i, pred in enumerate(st.session_state.predicates):
        if st.session_state.editing_pred == ('simple', i):
            render_simple_edit_form(i, pred)
            continue
        col_desc, col_edit, col_del = st.columns([5, 1, 1])
        col_desc.markdown(f"**{pred['name']}**: {describe_predicate(pred)}")
        col_edit.button("Edit", key=f"edit_pred_{i}",
                        on_click=start_pred_edit, args=('simple', i))
        col_del.button("Remove", key=f"del_pred_{i}",
                       on_click=remove_predicate, args=('simple', i))
    for i, comp in enumerate(st.session_state.composite_predicates):
        if st.session_state.editing_pred == ('composite', i):
            render_composite_edit_form(i, comp)
            continue
        col_desc, col_edit, col_del = st.columns([5, 1, 1])
        col_desc.markdown(f"**{comp['name']}**: {comp['expression']} *(composite)*")
        col_edit.button("Edit", key=f"edit_comp_{i}",
                        on_click=start_pred_edit, args=('composite', i))
        col_del.button("Remove", key=f"del_comp_{i}",
                       on_click=remove_predicate, args=('composite', i))

    st.divider()

    # ---- Unified add-predicate form ----
    st.subheader("Add Predicate")

    if st.session_state.processed_df is None:
        st.warning("Please load and apply a dataset first to create predicates.")
    else:
        pred_kind = st.radio(
            "Predicate type",
            ["Simple", "Composite"],
            horizontal=True,
            help="Simple predicates test a single feature; composite predicates "
                 "combine two already-defined predicates with AND/OR/NOT."
        )

        if pred_kind == "Simple":
            columns, boolean_columns = get_feature_columns()
            numeric_columns = set(col for col in columns if col not in boolean_columns)

            # Show info about column types
            if boolean_columns and numeric_columns:
                st.info(f"**Boolean features:** {len(boolean_columns)} | **Numeric features:** {len(numeric_columns)}")

            # Select the column first (outside the form) so the fields can adapt
            selected_column = st.selectbox(
                "Select Feature Column",
                columns,
                key="predicate_column_select",
                help="Select a column to create a predicate for"
            )

            is_boolean = selected_column in boolean_columns
            column_index = columns.index(selected_column) if selected_column in columns else 0

            if is_boolean:
                # Boolean Predicate Form (simplified)
                st.caption(f"📌 **{selected_column}** is a boolean feature (values: 0 or 1)")
                with st.form("simple_predicate_form"):
                    # Auto-fill predicate name with column name
                    pred_name = st.text_input(
                        "Predicate Name",
                        value=selected_column,
                        key="bool_pred_name",
                        help="Name for this predicate (auto-filled with column name)"
                    )

                    submit_pred = st.form_submit_button("Add Predicate")

                    if submit_pred and pred_name and not(name_exists(pred_name)):
                        # Under the hood: boolean True = "> 0.5"
                        st.session_state.predicates.append({
                            'name': pred_name,
                            'column_index': column_index,
                            'threshold': 0.5,
                            'comparison': 'Greater',
                            'is_boolean': True,
                            'column_name': selected_column
                        })
                        mark_predicates_dirty()
                        logging_service.append_event(logging_service.make_event(
                            "DomainExpert", "human",
                            f"defined predicate: {pred_name} (feature={selected_column}, boolean=True)",
                            event_type="predicate_definition"
                        ))
                        st.rerun()
                    elif submit_pred and name_exists(pred_name):
                        st.warning(f"A predicate named '{pred_name}' already exists.")
            else:
                # Numeric Predicate Form (with threshold and comparison)
                st.caption(f"**{selected_column}** is a numeric feature")
                with st.form("simple_predicate_form"):
                    # Auto-fill predicate name with column name
                    pred_name = st.text_input(
                        "Predicate Name",
                        value=selected_column,
                        key="num_pred_name",
                        help="Name for this predicate (auto-filled with column name)"
                    )
                    threshold = st.number_input("Threshold Value", format="%f")
                    comparison = st.selectbox("Comparison Operator", ["Greater", "Less", "Equal"])

                    submit_pred = st.form_submit_button("Add Predicate")

                    if submit_pred and pred_name and not(name_exists(pred_name)):
                        st.session_state.predicates.append({
                            'name': pred_name,
                            'column_index': column_index,
                            'threshold': threshold,
                            'comparison': comparison,
                            'is_boolean': False,
                            'column_name': selected_column
                        })
                        mark_predicates_dirty()
                        logging_service.append_event(logging_service.make_event(
                            "DomainExpert", "human",
                            f"defined predicate: {pred_name} (feature={selected_column}, operator={comparison}, threshold={threshold})",
                            event_type="predicate_definition"
                        ))
                        st.rerun()
                    elif submit_pred and name_exists(pred_name):
                        st.warning(f"A predicate named '{pred_name}' already exists.")

        elif pred_kind == "Composite":
            predicate_names = [pred['name'] for pred in st.session_state.predicates]

            if not predicate_names:
                st.warning("Please define at least one simple predicate first.")
            else:
                with st.form("composite_predicate_form"):
                    comp_pred_name = st.text_input("Composite Predicate Name")

                    col1, col2 = st.columns(2)

                    with col1:
                        left_pred = st.selectbox("Left Predicate", predicate_names)
                        left_unary = st.selectbox("Left Unary Operator", ["No Unary Predicate", "NOT"])

                    with col2:
                        right_pred = st.selectbox("Right Predicate", predicate_names)
                        right_unary = st.selectbox("Right Unary Operator", ["No Unary Predicate", "NOT"])

                    binary_op = st.selectbox("Binary Operator", ["AND", "OR"])

                    submit_composite = st.form_submit_button("Add Predicate")

                    if submit_composite and comp_pred_name and not(name_exists(comp_pred_name)):
                        left_expr = f"Not({left_pred})" if left_unary == "NOT" else left_pred
                        right_expr = f"Not({right_pred})" if right_unary == "NOT" else right_pred
                        expression = f"{binary_op.capitalize()}({left_expr}, {right_expr})"
                        st.session_state.composite_predicates.append({
                            'name': comp_pred_name,
                            'expression': expression
                        })
                        mark_predicates_dirty()
                        logging_service.append_event(logging_service.make_event(
                            "DomainExpert", "human",
                            f"defined composite predicate: {comp_pred_name} = {expression}",
                            event_type="predicate_definition"
                        ))
                        st.rerun()
                    elif submit_composite and name_exists(comp_pred_name):
                        st.warning(f"A predicate or composite predicate named '{comp_pred_name}' already exists. Please choose another name.")

    # ---- Rules section (same tab, so predicates and rules can be managed together) ----
    st.divider()
    st.subheader("Define Rules")

    has_predicates = bool(st.session_state.predicates or st.session_state.composite_predicates)
    if not has_predicates:
        st.info("👆 Please define at least one predicate above first.")
    else:
        rule_predicate_names = collect_predicate_names(
            st.session_state.predicates,
            st.session_state.composite_predicates,
            st.session_state.target_column,
        )

        rule_text = st.text_input(
            "Enter rule in natural language:",
            value="",
            help="Example: 'if the clump thickness is high but mitoses is low "
                 "then the tumor is malignant'. Reference the predicates you defined."
        )

        if st.button("Add Rule"):
            if not rule_text.strip():
                st.warning("Please enter a rule first.")
            else:
                _t0 = time.time()
                try:
                    with st.spinner("Interpreting rule…"):
                        structured = predicate_service.parse_rule_text(rule_text, rule_predicate_names)
                except Exception as e:
                    st.error(f"Could not interpret the rule: {e}")
                else:
                    _rule_parse_latency_ms = round((time.time() - _t0) * 1000, 1)
                    st.session_state.rules.append({"text": rule_text, "structured": structured})
                    st.session_state.rules_saved = False
                    logging_service.append_event(logging_service.make_event(
                        "DomainExpert", "human",
                        f"authored rule: {rule_text}",
                        event_type="rule_authoring"
                    ))
                    logging_service.append_event(logging_service.make_event(
                        "RuleParser_AI", "ai",
                        f"parsed natural language rule to LTN formula: {canonical.rule_to_text(structured)}",
                        latency_ms=_rule_parse_latency_ms,
                        correct=True,
                        event_type="decision"
                    ))
                    st.rerun()

        def remove_rule(index):
            removed = st.session_state.rules.pop(index)
            st.session_state.editing_rule = None
            st.session_state.rules_saved = False
            logging_service.append_event(logging_service.make_event(
                "DomainExpert", "human",
                f"removed rule: {removed['text']}",
                event_type="rule_removal"
            ))

        def render_rule_edit_form(index, rule):
            with st.container(border=True):
                st.markdown("Editing rule")
                new_text = st.text_input("Rule in natural language",
                                         value=rule['text'], key=f"edit_rule_text_{index}")
                st.caption(f"Currently understood as: `{canonical.rule_to_text(rule['structured'])}`")
                col_save, col_cancel = st.columns([1, 1])
                # Saving re-runs the LLM so the encoded form always matches the text.
                if col_save.button("Re-interpret & Save", key=f"edit_rule_save_{index}", type="primary"):
                    if not new_text.strip():
                        st.warning("The rule text cannot be empty.")
                    else:
                        _t0 = time.time()
                        try:
                            with st.spinner("Interpreting rule…"):
                                structured = predicate_service.parse_rule_text(
                                    new_text, rule_predicate_names)
                        except Exception as e:
                            st.error(f"Could not interpret the rule: {e}")
                        else:
                            _latency_ms = round((time.time() - _t0) * 1000, 1)
                            st.session_state.rules[index] = {"text": new_text, "structured": structured}
                            st.session_state.editing_rule = None
                            st.session_state.rules_saved = False
                            logging_service.append_event(logging_service.make_event(
                                "DomainExpert", "human",
                                f"edited rule: {rule['text']} -> {new_text}",
                                event_type="rule_edit"
                            ))
                            logging_service.append_event(logging_service.make_event(
                                "RuleParser_AI", "ai",
                                f"parsed natural language rule to LTN formula: {canonical.rule_to_text(structured)}",
                                latency_ms=_latency_ms,
                                correct=True,
                                event_type="decision"
                            ))
                            st.rerun()
                col_cancel.button("Cancel", key=f"edit_rule_cancel_{index}",
                                  on_click=cancel_rule_edit)

        # ---- Always-visible list of defined rules ----
        st.subheader("Defined Rules")
        if not st.session_state.rules:
            st.caption("No rules defined yet — describe one above in plain language.")
        for i, rule in enumerate(st.session_state.rules):
            if st.session_state.editing_rule == i:
                render_rule_edit_form(i, rule)
                continue
            col_desc, col_edit, col_del = st.columns([5, 1, 1])
            with col_desc:
                st.markdown(f"**{rule['text']}**")
                # The encoded form lets the user verify what the AI understood.
                st.caption(f"Understood as: `{canonical.rule_to_text(rule['structured'])}`")
            col_edit.button("Edit", key=f"edit_rule_{i}", on_click=start_rule_edit, args=(i,))
            col_del.button("Remove", key=f"del_rule_{i}", on_click=remove_rule, args=(i,))

    # ---- Single finalization point: writes every predicate & rule artifact ----
    if has_predicates:
        st.divider()
        has_unsaved = (not st.session_state.predicates_saved
                       or (st.session_state.rules and not st.session_state.rules_saved))
        if has_unsaved:
            st.info("You have unsaved changes — press Save when you are done editing.")
        if st.button("Save Predicates & Rules", type="primary",
                     disabled=st.session_state.target_column is None):
            predicate_service.generate_and_save_predicates(
                st.session_state.target_column,
                st.session_state.predicates,
                st.session_state.composite_predicates
            )
            st.session_state.predicates_saved = True
            if st.session_state.rules:
                predicate_service.save_rules_artifacts(
                    st.session_state.predicates,
                    st.session_state.composite_predicates,
                    st.session_state.rules,
                )
                st.session_state.rules_saved = True
            logging_service.append_event(logging_service.make_event(
                "DomainExpert", "human",
                f"saved {len(st.session_state.predicates)} predicate(s), "
                f"{len(st.session_state.composite_predicates)} composite(s) and "
                f"{len(st.session_state.rules)} rule(s)",
                event_type="predicate_definition"
            ))
            if st.session_state.rules:
                st.success("Predicates & rules saved — continue on the Train tab.")
            else:
                st.success("Predicates saved — now define your rules below.")

    st.divider()

    # ---- Export predicates & rules to a file ----
    with st.expander("Export predicates & rules to a file"):
        export_rule_set = predicate_service.load_rule_set()
        if not export_rule_set or not (
            export_rule_set.get("predicates") or export_rule_set.get("rules")
        ):
            st.caption(
                "Define or import predicates & rules first (and save your rules), "
                "then export them here in any supported format."
            )
        else:
            export_format = st.selectbox(
                "Export format",
                options=list(rule_exporter.SUPPORTED_FORMATS.keys()),
                format_func=lambda ext: rule_exporter.SUPPORTED_FORMATS[ext],
                key="rules_export_format",
            )
            try:
                export_text = rule_exporter.export_rule_set(export_rule_set, export_format)
                st.code(export_text)
                st.download_button(
                    "Download rules file",
                    data=export_text,
                    file_name=f"rules.{export_format}",
                    mime="application/json" if export_format == "json" else "text/plain",
                    key="rules_export_download",
                )
            except Exception as e:
                st.error(f"Failed to export rules: {e}")

with tab3:
    st.header("Train Models")

    # Training is gated on a finalized rule set from the Predicates & Rules tab.
    if st.session_state.get('rules_saved', False):
        st.subheader("Step 1: Choose Training Method")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🤖 Automated Training")
            st.write("Train LTN and MLP models directly in Streamlit with the default pipeline.")
            run = st.button("Train Models", type="primary", use_container_width=True)
            if run:
                logging_service.append_event(logging_service.make_event(
                    "DomainExpert", "human", "triggered model training",
                    event_type="info"
                ))
                with st.spinner("Training models..."):
                    model_service.run_training_pipeline(GLOBAL_SEED, TRAIN_EPOCHS)
                    st.session_state.models_trained = True
                    st.rerun()

        with col2:
            st.markdown("### 📓 Custom Training in JupyterLab")
            st.write("Open JupyterLab with a pre-configured notebook. Modify hyperparameters, architecture, or training logic as needed.")

            # Session notebook info
            session_notebook_name = "ltn_training_session.ipynb"

            # Open JupyterLab button
            jupyter_url = notebook_service.get_jupyter_notebook_url(session_notebook_name)
            st.link_button("Open JupyterLab", jupyter_url, type="primary", use_container_width=True)

            # Show download button
            if notebook_service.session_notebook_exists():
                session_notebook_path = notebook_service.get_session_notebook_path()
                notebook_content = notebook_service.get_notebook_download_content(session_notebook_path)
                st.download_button(
                    label="📥 Download Notebook",
                    data=notebook_content,
                    file_name=session_notebook_name,
                    mime="application/x-ipynb+json",
                    use_container_width=True
                )
            else:
                st.warning("Session notebook not found in notebooks directory")

        st.divider()

        # Step 2: Unified Model Evaluation
        st.subheader("Step 2: Model Evaluation")

        available_models = model_service.get_available_models()
        if available_models:
            st.caption(f"Available models: {', '.join([m['name'] for m in available_models])}")

            evaluate = st.button("Evaluate Models", type="secondary", use_container_width=False)
            if evaluate or st.session_state.get('models_trained', False):
                st.session_state.models_trained = False  # Reset flag
                with st.spinner("Evaluating models..."):
                    results = model_service.evaluate_available_models(
                        st.session_state.X_test,
                        st.session_state.y_test
                    )
                if results:
                    for res_name in ['Accuracy', 'AUROC', 'F1', 'Precision', 'Recall']:
                        if any(res_name in r for r in results.values()):
                            create_metric_expander(results, res_name, res_name == 'Accuracy')
                    _ltn_f1 = results.get('LTN', {}).get('F1')
                    _mlp_f1 = results.get('MLP', {}).get('F1')
                    _rules_f1 = results.get('RULES', {}).get('F1')
                    _ltn_outperforms = (
                        _ltn_f1 is not None and
                        (_mlp_f1 is None or _ltn_f1 > _mlp_f1) and
                        (_rules_f1 is None or _ltn_f1 > _rules_f1)
                    )
                    logging_service.append_event(logging_service.make_event(
                        "ModelEvaluator_AI", "ai",
                        f"evaluated model performance: LTN (F1={_ltn_f1}) vs MLP (F1={_mlp_f1}) vs Rules (F1={_rules_f1})",
                        correct=_ltn_outperforms,
                        probs={k: v.get('F1') for k, v in results.items() if v.get('F1') is not None},
                        event_type="ltn_outperforms"
                    ))
                    logging_service.publish_events(
                        st.session_state.get("minio_token"),
                        MINIO_BUCKET,
                        st.session_state.run_id
                    )
                    st.success("Models ready! Use the Explain tab to make predictions.")
                else:
                    st.warning("No models could be evaluated.")
        else:
            st.info("No trained models found. Use one of the training methods above.")
    else:
        st.info("👆 Please define and save your rules on the Predicates & Rules tab first to proceed with training.")

with tab4:
    if st.session_state.X_test is not None:
        # Check for available models
        available_models = model_service.get_available_models()
        if available_models:
            st.caption(f"Available models: {', '.join([m['name'] for m in available_models])}")

        # Use domain-appropriate terminology based on dataset description
        sample_label = "Sample"
        if st.session_state.dataset_description:
            row_desc = st.session_state.dataset_description.get('row_description', '')
            if row_desc:
                # Extract a short label from the row description
                sample_label = row_desc.split()[0] if row_desc else "Sample"
                if sample_label.lower() in ['a', 'an', 'the']:
                    sample_label = row_desc.split()[1] if len(row_desc.split()) > 1 else "Sample"

        sample_id = st.number_input(f"Enter {sample_label} Id:", 0, st.session_state.X_test.shape[0] - 1, 0, placeholder=f"{sample_label} Id")
        x = st.session_state.X_test[sample_id]
        y = st.session_state.y_test[sample_id]

        # Display original image or text if available
        if st.session_state.test_indices is not None:
            original_idx = st.session_state.test_indices[sample_id]

            # Display original image
            if st.session_state.dataset_type == "image" and st.session_state.original_image_paths is not None:
                st.subheader('Original Image:')
                img_path = st.session_state.original_image_paths[original_idx]
                # Reconstruct full path using the extracted dataset path
                if st.session_state.extracted_dataset_path:
                    full_path = os.path.join(st.session_state.extracted_dataset_path, img_path)
                    if os.path.exists(full_path):
                        img = Image.open(full_path)
                        st.image(img, width=300)
                    else:
                        st.warning(f"Image not found: {full_path}")

            # Display original text
            elif st.session_state.dataset_type == "text" and st.session_state.original_texts is not None:
                st.subheader('Original Text:')
                original_text = st.session_state.original_texts[original_idx]
                st.text_area("", value=original_text, height=150, disabled=True, label_visibility="collapsed")

        table_contents = pd.DataFrame(
            dict(zip(data_service.get_feature_names(), st.session_state.X_test[sample_id])),
            index=[0]
        ).T.reset_index()
        table_contents.columns = ['Feature', 'Value']

        st.subheader('Feature Values:')
        st.table(table_contents.style.hide(axis='index'))
        st.markdown(f'**True label:** {y.item()}')
        predict = st.button("Predict!")
        if predict:
            interaction_id = f"diabetes_case_sample{sample_id}"
            logging_service.append_event(logging_service.make_event(
                "DomainExpert", "human",
                f"triggered prediction for sample_id={sample_id}",
                interaction_id=interaction_id,
                event_type="info"
            ))
            explanation = explanation_service.predict_and_explain(
                x, y,
                st.session_state.X_train,
                st.session_state.target_column,
                [r["text"] for r in st.session_state.rules],
                dataset_description=st.session_state.dataset_description
            )

            _scores = explanation['scores']
            _ltn_score = _scores.get('LTN', 0.5)
            _predicted_class = explanation_service.get_predicted_class(
                _ltn_score, st.session_state.target_column
            )
            logging_service.append_event(logging_service.make_event(
                "LTN_Classifier_AI", "ai",
                f"classifying sample_id={sample_id}",
                interaction_id=interaction_id,
                latency_ms=explanation['prediction_latency_ms'],
                duration_s=round(explanation['prediction_latency_ms'] / 1000, 3),
                probs={k: round(float(v), 4) for k, v in _scores.items()},
                event_type="decision"
            ))
            logging_service.append_event(logging_service.make_event(
                "SHAP_Explainer_AI", "ai",
                f"generated feature importance (top-3: {', '.join(explanation['important_features'])})",
                interaction_id=interaction_id,
                event_type="info"
            ))
            logging_service.append_event(logging_service.make_event(
                "XAI_RAG_AI", "ai",
                f"generated natural language explanation for prediction: {_predicted_class}",
                interaction_id=interaction_id,
                latency_ms=explanation['rag_latency_ms'],
                duration_s=round(explanation['rag_latency_ms'] / 1000, 3),
                event_type="decision"
            ))

            # Display predictions
            table_contents = pd.DataFrame(explanation['scores'], index=['value']).transpose()
            st.subheader('Predictions:')
            st.table(table_contents)

            # Display feature importance
            st.subheader('Feature Importance:')
            decision_plot(explanation['shap_values'], x)

            # Display satisfied rules
            if explanation['satisfied_rules']:
                st.subheader('Rules Triggered:')
                for rule in explanation['satisfied_rules']:
                    st.write(rule)

            # Display RAG explanation
            st.subheader('Summary:')
            st.markdown(explanation['rag_explanation'])

            logging_service.append_event(logging_service.make_event(
                "DomainExpert", "human",
                f"reviewed AI prediction and explanation for sample_id={sample_id}",
                interaction_id=interaction_id,
                correct=True,
                event_type="decision"
            ))
    else:
        st.warning("Please upload a dataset to get predictions.")

with tab5:
    st.header("Published Models")
    token = st.session_state.get("minio_token")
    if not token:
        st.warning("MinIO not authenticated. Please upload a .env file.")
    else:
        if st.button("Refresh", key="models_refresh"):
            st.rerun()

        # Current production model banner
        prod_manifest = minio_utils.minio_read_json(token, MINIO_BUCKET, "production/manifest.json")
        if prod_manifest:
            st.success(
                f"Current production model: version **{prod_manifest['version']}** "
                f"— promoted {prod_manifest['promoted_at']}"
            )
        else:
            st.info("No model in production yet.")

        st.divider()

        index = minio_utils.minio_read_json(token, MINIO_BUCKET, "published-models/index.json")
        if not index:
            st.info("No models published yet.")
        else:
            promoted_this_render = False
            header_col1, header_col2, header_col3 = st.columns([2, 3, 2])
            header_col1.markdown("**Version**")
            header_col2.markdown("**Published at**")
            header_col3.markdown("**Status**")
            st.divider()

            for entry in index:
                version = entry["version"]
                status = entry["status"]

                if status == "pending_approval":
                    if check_and_promote(token, version):
                        status = "production"
                        promoted_this_render = True

                col1, col2, col3 = st.columns([2, 3, 2])
                col1.write(version)
                col2.write(entry.get("published_at", "—"))
                with col3:
                    if status == "production":
                        st.success("production")
                    else:
                        st.warning("pending approval")

            if promoted_this_render:
                st.rerun()

