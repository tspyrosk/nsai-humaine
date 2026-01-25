import os
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
import shap
import absl.logging
import dataset.minio_utils as minio_utils
from utils import remove_rules_and_predicates, remove_outputs, remove_rules
from dotenv import dotenv_values
import io
from paths import *

# Import business logic services
from services import data_service, model_service, explanation_service, predicate_service, image_service
from PIL import Image

absl.logging.set_verbosity(absl.logging.ERROR)

st.set_page_config(layout="wide")

GLOBAL_SEED = st.sidebar.number_input("Enter a Seed for the current run:", 0, 10000, 99, placeholder="Run Seed")

SAMPLE_RATIO = st.sidebar.number_input("Data Ratio: ", value=1.0, min_value=0.1, max_value=1.0, step=0.1)
TRAIN_EPOCHS = st.sidebar.number_input("Training Epochs:", 1, 2000, 40)

env_file = st.sidebar.file_uploader("Upload .env file", type=["env"])

if env_file:
    env_text = env_file.getvalue().decode("utf-8")
    env_vars = dotenv_values(stream=io.StringIO(env_text))
    for key, value in env_vars.items():
        os.environ[key] = value
    st.sidebar.success(".env loaded from memory")
    minio_utils.TOKEN = minio_utils.minio_auth(os.getenv("MINIO_USER"), os.getenv("MINIO_PASS"))
    st.sidebar.success("Obtained MinIO token")

def init_predicates_state():
    if 'predicates' not in st.session_state:
        st.session_state.predicates = []
    if 'composite_predicates' not in st.session_state:
        st.session_state.composite_predicates = []

def reset_predicates_state():
    st.session_state.predicates = []
    st.session_state.composite_predicates = []

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
    st.session_state.rules = []
    reset_predicates_state()
    st.cache_data.clear()

if st.sidebar.button("Reset Rules"):
    remove_rules()
    st.session_state.rules = []
    #st.success("Previous rules deleted.")
if st.sidebar.button("Reset Rules and Predicates"):
    remove_rules_and_predicates()
    st.session_state.rules = []
    reset_predicates_state()
    #st.success("Rules and predicates deleted. Please start again!")
if st.sidebar.button("Reset All"):
    remove_outputs()
    reset_state()
    st.rerun()


init_state()

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


tab1, tab2, tab3, tab4 = st.tabs(["Load", "Predicates", "Train", "Explain"])

with tab1:
    st.header("Upload Dataset")

    upload_method = st.radio("Choose data source", ["Upload CSV", "MinIO Path", "Image Dataset"])

    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            st.session_state.df = data_service.save_uploaded_csv(
                uploaded_file.getbuffer(),
                INPUT_CSV
            )
            st.session_state.dataset_type = "csv"

    elif upload_method == "MinIO Path":
        minio_path = st.text_input("Enter MinIO Path")
        if st.button("Load from MinIO"):
            if minio_path:
                try:
                    bucket = minio_path.split("/")[0]
                    path = "/".join(minio_path.split("/")[1:])
                    print(f"Reading from MinIO. Bucket: {bucket}, rePath: {path}")
                    minio_utils.minio_download(minio_utils.TOKEN, bucket, path, INPUT_CSV)
                    st.session_state.df = pd.read_csv(INPUT_CSV)
                    st.session_state.dataset_type = "csv"
                except Exception as e:
                    st.error(f"Error loading data from MinIO: {str(e)}")
            else:
                st.warning("Please enter a valid MinIO path.")

    else:  # Image Dataset
        st.subheader("Load Image Dataset")
        st.info("Upload a ZIP file containing two folders (one for each class). "
                "Example structure: `dataset.zip` → `class_a/` and `class_b/` folders with images inside.")

        uploaded_zip = st.file_uploader("Upload ZIP file with image dataset", type=["zip"])

        if uploaded_zip is not None:
            # Create a persistent directory for the extracted dataset
            dataset_name = os.path.splitext(uploaded_zip.name)[0]
            dataset_path = os.path.join(INPUT_DIR, f"uploaded_{dataset_name}")

            # Check if we need to extract (avoid re-extracting on every rerun)
            if 'extracted_dataset_path' not in st.session_state or st.session_state.extracted_dataset_path != dataset_path:
                # Clean up previous extraction if exists
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)

                # Extract ZIP file
                with st.spinner("Extracting ZIP file..."):
                    os.makedirs(dataset_path, exist_ok=True)
                    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()), 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)

                    # Handle case where ZIP contains a single root folder
                    extracted_items = [f for f in os.listdir(dataset_path) if not f.startswith('.')]
                    if len(extracted_items) == 1:
                        single_folder = os.path.join(dataset_path, extracted_items[0])
                        if os.path.isdir(single_folder):
                            # Move contents up one level
                            for item in os.listdir(single_folder):
                                shutil.move(os.path.join(single_folder, item), dataset_path)
                            os.rmdir(single_folder)

                st.session_state.extracted_dataset_path = dataset_path
                st.success(f"ZIP extracted to: {dataset_path}")

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
                                feature_progress_callback=lambda c, t: update_progress(c, t, "Extracting image features")
                            )

                            # Save as CSV for compatibility with existing data flow
                            df.to_csv(INPUT_CSV, index=False)

                            # Store in session state
                            st.session_state.df = df
                            st.session_state.image_metadata = metadata
                            st.session_state.dataset_type = "image"

                            progress_bar.progress(1.0)
                            progress_text.text("Processing complete!")

                            st.success(f"Image dataset converted to tabular format! "
                                      f"({len(metadata['all_tags'])} tag features + "
                                      f"{len(metadata['image_feature_names'])} image features)")

                        except Exception as e:
                            st.error(f"Error processing image dataset: {str(e)}")

            except Exception as e:
                st.error(f"Error loading image dataset: {str(e)}")

    # Common section for all data sources - display DataFrame and configure
    if st.session_state.df is not None:
        st.divider()
        st.subheader("Dataset Preview")
        st.write(st.session_state.df.head())

        # For image datasets, show metadata
        if st.session_state.dataset_type == "image" and st.session_state.image_metadata:
            meta = st.session_state.image_metadata
            with st.expander("Image Dataset Details"):
                st.write(f"**Tag features:** {len(meta['all_tags'])}")
                st.write(f"**Image features:** {len(meta['image_feature_names'])}")
                st.write(f"**Sample tags:** {', '.join(meta['all_tags'][:10])}...")

        columns = st.session_state.df.columns.tolist()

        # Filter out non-feature columns for dropping
        droppable_columns = [c for c in columns if c not in ['image_path']]
        columns_to_drop = st.multiselect("Select columns to drop", droppable_columns)

        # For image datasets, default target to 'label'
        default_target_idx = columns.index('label') if 'label' in columns else 0
        st.session_state.target_column = st.selectbox("Select target column", columns, index=default_target_idx)

        if st.button("Apply"):
            # Always drop image_path if it exists (not a feature)
            cols_to_drop = columns_to_drop.copy()
            if 'image_path' in columns and 'image_path' not in cols_to_drop:
                cols_to_drop.append('image_path')

            X_train, X_test, y_test, processed_df = data_service.load_data_from_csv(
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

            st.success("Train and test sets loaded into session state!")


with tab2:
    st.header("Define Predicates")

    def display_predicates_and_generate_code():
        st.subheader("Defined Predicates")
        if st.session_state.predicates:
            st.write("Simple Predicates:")
            for pred in st.session_state.predicates:
                st.write(f"{pred['name']}: Column {pred['column_index']}, {pred['comparison']} {pred['threshold']}")

        if st.session_state.composite_predicates:
            st.write("Composite Predicates:")
            for pred in st.session_state.composite_predicates:
                st.write(f"{pred['name']}: {pred['expression']}")

        if st.session_state.predicates or st.session_state.composite_predicates:
            predicate_code, lambda_code, predicate_names = predicate_service.generate_and_save_predicates(
                st.session_state.target_column,
                st.session_state.predicates,
                st.session_state.composite_predicates
            )
            st.subheader("Generated Code")
            st.code(predicate_code + "\n" + lambda_code)

    def name_exists(name):
        return predicate_service.check_predicate_name_exists(
            name,
            st.session_state.predicates,
            st.session_state.composite_predicates
        )
    
    # Simple Predicate Form
    st.subheader("Create Simple Predicate")
    with st.form("simple_predicate_form"):
        pred_name = st.text_input("Predicate Name")
        if st.session_state.processed_df is not None:
            columns = st.session_state.processed_df.columns.tolist()
            column = st.selectbox("Select Column", columns)
            column_index = columns.index(column) if column in columns else 0
        else:
            column_index = st.number_input("Column Index", min_value=0, step=1)
        threshold = st.number_input("Threshold Value", format="%f")
        comparison = st.selectbox("Comparison Operator", ["Greater", "Less", "Equal"])
        submit_simple = st.form_submit_button("Add Predicate")
        
        if submit_simple and pred_name and not(name_exists(pred_name)):
            st.session_state.predicates.append({
                'name': pred_name,
                'column_index': column_index,
                'threshold': threshold,
                'comparison': comparison
            })
            #st.success(f"Predicate {pred_name} added!")
            display_predicates_and_generate_code()
        elif name_exists(pred_name):
            st.warning(f"A predicate or composite predicate named '{pred_name}' already exists. Please choose another name.")


    # Composite Predicate Form
    st.subheader("Create Composite Predicate")
    with st.form("composite_predicate_form"):
        comp_pred_name = st.text_input("Composite Predicate Name")
        
        # Get list of existing predicate names
        predicate_names = [pred['name'] for pred in st.session_state.predicates]
        
        if predicate_names:
            col1, col2 = st.columns(2)
            
            with col1:
                left_pred = st.selectbox("Left Predicate", predicate_names)
                left_unary = st.selectbox("Left Unary Operator", ["No Unary Predicate", "NOT"])
            
            with col2:
                right_pred = st.selectbox("Right Predicate", predicate_names)
                right_unary = st.selectbox("Right Unary Operator", ["No Unary Predicate", "NOT"])
            
            binary_op = st.selectbox("Binary Operator", ["AND", "OR"])
            
            # Construct expression
            left_expr = f"Not({left_pred})" if left_unary == "NOT" else left_pred
            right_expr = f"Not({right_pred})" if right_unary == "NOT" else right_pred
            expression = f"{binary_op.capitalize()}({left_expr}, {right_expr})"
            
            st.text_input("Generated Expression", value=expression, disabled=True)
        else:
            st.warning("Please define at least one simple predicate first.")
            expression = ""
        
        submit_composite = st.form_submit_button("Add Composite Predicate")
        
        if submit_composite and comp_pred_name and predicate_names and not(name_exists(comp_pred_name)):
            st.session_state.composite_predicates.append({
                'name': comp_pred_name,
                'expression': expression
            })
            #st.success(f"Composite Predicate {comp_pred_name} added!") 
            display_predicates_and_generate_code()   
        elif name_exists(comp_pred_name):
            st.warning(f"A predicate or composite predicate named '{pred_name}' already exists. Please choose another name.")

with tab3:
    rule_text = st.text_input("Enter rule:", value="")
    if 'rules' not in st.session_state:
        st.session_state.rules = []

    add_rule = st.button("Add Rule")

    if add_rule:
        st.session_state.rules.append(rule_text)

    selected_rules = st.pills("Rules Entered:", st.session_state.rules, selection_mode="multi")

    save_rules = st.button("Save")
    if save_rules:
        ltn_code, rules_code = predicate_service.save_and_parse_rules(selected_rules)
        st.subheader("Generated Code")
        st.code(ltn_code + "\n" + rules_code)

    run = st.button("Train Models")
    if run:
        results, models = model_service.run_all_experiments(
            st.session_state.X_test,
            st.session_state.y_test,
            GLOBAL_SEED,
            TRAIN_EPOCHS
        )
        for (res_name, res_value) in results['MLP'].items():
            create_metric_expander(results, res_name, res_name == 'Accuracy')

with tab4:
    if st.session_state.X_test is not None:
        patient_id = st.number_input("Enter Patient Id:", 0, st.session_state.X_test.shape[0] - 1, 0, placeholder="Patient Id")
        x = st.session_state.X_test[patient_id]
        y = st.session_state.y_test[patient_id]

        table_contents = pd.DataFrame(
            dict(zip(data_service.get_feature_names(), st.session_state.X_test[patient_id])),
            index=[0]
        ).T.reset_index()
        table_contents.columns = ['Feature', 'Value']

        st.subheader('Feature Values:')
        st.table(table_contents.style.hide(axis='index'))
        st.markdown(f'**True label:** {y.item()}')
        predict = st.button("Predict!")
        if predict:
            explanation = explanation_service.predict_and_explain(
                x, y,
                st.session_state.X_train,
                st.session_state.target_column,
                selected_rules
            )

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
    else:
        st.warning("Please upload a dataset to get predictions.")

