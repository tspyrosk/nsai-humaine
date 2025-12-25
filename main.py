import os
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import shap
import absl.logging
import os
import dataset.minio_utils as minio_utils
from predicates_definition.predicate_gen_utils import generate_python_code
from utils import collect_predicate_names, remove_rules_and_predicates, remove_outputs, remove_rules
from dotenv import dotenv_values
import io
from paths import *

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
    init_predicates_state()
    st.cache_data.clear()

def reset_state():
    st.session_state.df = None
    st.session_state.target_column = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.processed_df = None
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

@st.cache_data
def load_data(target, cols_to_drop):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    drop = ",".join(cols_to_drop)
    os.system(f"python {SPLIT_DATASET_SCRIPT} --input_file_path={INPUT_CSV} --output_path={OUTPUT_DIR} --seed={GLOBAL_SEED} --sample_ratio={SAMPLE_RATIO} --target_column=\"{target}\" --drop=\"{drop}\"")
    return True


def get_feature_names():
    path = FEATURE_NAMES_PATH
    if os.path.exists(path):
        with open(path, 'r') as f:
            loaded_features = [line.strip() for line in f]
            return loaded_features
    return []

def get_predicted_class(pred):
    if pred>0.5:
        cl = st.session_state.target_column.lower()
    else:
        cl = f"no {st.session_state.target_column.lower()}"
    return cl

def run_training_pipeline():
    os.system(f"python {BASE_DIR}/models/train_ltn.py --input_path={OUTPUT_DIR} --seed={GLOBAL_SEED} --use_rules=1 --rules_file_path={LTN_RULES_PATH} --epochs={TRAIN_EPOCHS}")
    os.system(f"python {BASE_DIR}/models/train_ltn.py --input_path={OUTPUT_DIR} --seed={GLOBAL_SEED} --epochs={TRAIN_EPOCHS}")

class RulesModel:
    def __init__(self):
        self.name = "RULES"

    def score_samples(self, X, y):
        import models.create_rules_only as create_rules_only
        return create_rules_only.predict(X, y)

    def predict(self, X, y):
        preds = self.score_samples(X, y)
        return preds >= 0.5

class InferenceModel:
    def __init__(self, model_name):
        self.base_model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{model_name}.h5')
        self.name = model_name.upper()

    def score_samples(self, X, y):
        return self.base_model(X)

    def predict(self, X, y):
        preds = self.score_samples(X, y)
        return (preds > 0.5)

global models

def run_all_exps():
    all_shap_values = []
    results = {}

    run_training_pipeline()
    models = [
        RulesModel(),
        InferenceModel("mlp"),
        InferenceModel("ltn")
    ]

    for model in models:
        yp = model.predict(st.session_state.X_test, st.session_state.y_test)
        f1 = f1_score(st.session_state.y_test, yp)

        acc = balanced_accuracy_score(st.session_state.y_test, yp)
        auroc = roc_auc_score(st.session_state.y_test, model.score_samples(st.session_state.X_test, st.session_state.y_test), multi_class='ovr', average='weighted')
        report = classification_report(st.session_state.y_test, yp, output_dict=True)
        prec = report[str(1)]['precision']
        recall = report[str(1)]['recall']

        results[model.name] = {'Accuracy': acc,
                                     'AUROC': auroc,
                                     'F1': f1,
                                     'Precision': prec,
                                     'Recall': recall}
    return results


def explain_predictions(model_to_exp, x):
    def f(x):
        return model_to_exp.score_samples(x, np.zeros(x.shape[0]))

    explainer = shap.KernelExplainer(f, st.session_state.X_train)
    shap_values = explainer.shap_values(x)
    return shap_values


def predict_and_explain_random(x, y):
    trained_models = [
        RulesModel(),
        InferenceModel("mlp"),
        InferenceModel("ltn")
    ]

    collected_shap_values = []
    scores = {}
    for trained_model in trained_models:
        pred = trained_model.score_samples(np.expand_dims(x, axis=0), [y])
        scores[trained_model.name] = pred[0]
        shap_values = explain_predictions(trained_model, np.expand_dims(x, axis = 0))
        collected_shap_values.append(shap_values[0])
    import models.create_rules_only as create_rules_only
    rules_triggered = create_rules_only.get_satisfied_rule_indexes(x, y)
    concepts = create_rules_only.satisfied_concepts(x)
    important_features = top_3_shap_features(collected_shap_values[2])
    table_contents = pd.DataFrame(scores, index=['value']).transpose()
    st.subheader('Predictions:')
    st.table(table_contents)

    st.subheader('Feature Importance:')
    decision_plot(collected_shap_values[2].flatten(), x)

    if (len(rules_triggered) > 0) and (len(selected_rules) > 0):
        st.subheader('Rules Triggered:')
        for rule_idx in rules_triggered:
            st.write(selected_rules[rule_idx])

    st.subheader('Summary:')
    import models.xai_rag as xai_rag
    rag_explanation = xai_rag.extract_rag_explanation(concepts, important_features, get_predicted_class(scores["LTN"]))
    st.markdown(rag_explanation)

def top_3_shap_features(shap_values):
    top_3_idx = np.argsort(shap_values, axis=0)[-3:][::-1].tolist()
    return [get_feature_names()[idx[0]] for idx in top_3_idx]
def decision_plot(collected_shap_values, X_test):
    st_shap(shap.decision_plot(0.5, collected_shap_values, features=X_test, feature_names=get_feature_names()), height=500, width=800)


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
    
    upload_method = st.radio("Choose data source", ["Upload CSV", "MinIO Path"])
    
    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            os.makedirs(INPUT_DIR, exist_ok=True)
            with open(INPUT_CSV, "wb") as f:
                f.write(uploaded_file.getbuffer())  
            st.session_state.df = pd.read_csv(INPUT_CSV)  
    else:
        minio_path = st.text_input("Enter MinIO Path")
        if st.button("Load from MinIO"):
            if minio_path:
                try:
                    bucket = minio_path.split("/")[0]
                    path = "/".join(minio_path.split("/")[1:])
                    print(f"Reading from MinIO. Bucket: {bucket}, rePath: {path}")
                    minio_utils.minio_download(minio_utils.TOKEN, bucket, path, INPUT_CSV)
                    st.session_state.df = pd.read_csv(INPUT_CSV)
                except Exception as e:
                    st.error(f"Error loading data from MinIO: {str(e)}")
            else:
                st.warning("Please enter a valid MinIO path.")
    
    if st.session_state.df is not None:
        st.write(st.session_state.df.head())

        columns = st.session_state.df.columns.tolist()
        columns_to_drop = st.multiselect("Select columns to drop", columns)
        st.session_state.target_column = st.selectbox("Select target column", columns)
    
        if st.button("Apply"):
            loaded = load_data(st.session_state.target_column, columns_to_drop)
            st.session_state.X_train = np.load(f'{TRAIN_DATA_DIR}/X_{GLOBAL_SEED}.npy')
            st.session_state.X_test = np.load(f'{TEST_DATA_DIR}/X_{GLOBAL_SEED}.npy')
            st.session_state.y_test = np.load(f'{TEST_DATA_DIR}/y_{GLOBAL_SEED}.npy')
    
            st.success("Train and test sets loaded into session state!")
            
            with open(FEATURE_NAMES_PATH, 'r') as file:
                new_cols = [line.strip() for line in file if line.strip()]
            st.session_state.processed_df = pd.DataFrame(np.concatenate((st.session_state.X_train, st.session_state.X_test), axis=0), columns=new_cols)


with tab2:
    st.header("Define Predicates")

    def display_predicates_and_generate_code():
        st.subheader("Defined Predicates")
        if st.session_state.predicates:
            st.write("Simple Predicates:")
            #print(st.session_state.predicates)
            for pred in st.session_state.predicates:
                st.write(f"{pred['name']}: Column {pred['column_index']}, {pred['comparison']} {pred['threshold']}")
    
        if st.session_state.composite_predicates:
            st.write("Composite Predicates:")
            for pred in st.session_state.composite_predicates:
                st.write(f"{pred['name']}: {pred['expression']}")
        
        if st.session_state.predicates or st.session_state.composite_predicates:
            code = generate_python_code(st.session_state.target_column, st.session_state.predicates, st.session_state.composite_predicates)
            import predicates_definition.lambda_gen_utils as lambda_gen_utils
            rules_code =lambda_gen_utils.generate_python_code(st.session_state.target_column, st.session_state.predicates, st.session_state.composite_predicates)
            with open(f"{OUTPUT_DIR}/predicates.txt", "w") as file:
                file.write(code)
            with open(f"{OUTPUT_DIR}/lambdas.txt", "w") as file:
                file.write(rules_code)
            st.subheader("Generated Code")
            st.code(code + "\n" + rules_code)

            predicate_names = collect_predicate_names(st.session_state.predicates, st.session_state.composite_predicates, st.session_state.target_column)
            with open(f"{OUTPUT_DIR}/predicate_names.txt", "w") as file:
                file.write("\n".join(predicate_names))
    
    def name_exists(name):
        pred_names = [p['name'] for p in st.session_state.predicates]
        comp_pred_names = [c['name'] for c in st.session_state.composite_predicates]
        return name in pred_names or name in comp_pred_names
    
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
        with open(f"{OUTPUT_DIR}/rules_raw.txt", "w") as text_file:
            text_file.write("\n".join(selected_rules))
        os.system(f"python {TEXT2RULES_SCRIPT} --raw_rules_path={OUTPUT_DIR}/rules_raw.txt --output_path={OUTPUT_DIR}")
        with open(f"{OUTPUT_DIR}/rules_only_rules.txt", "r") as file:
            rules_code = file.read()
        with open(f"{OUTPUT_DIR}/ltn_rules.txt", "r") as file:
            ltn_code = file.read()
        st.subheader("Generated Code")
        st.code(ltn_code + "\n" + rules_code)

    run = st.button("Train Models")
    if run:
        results = run_all_exps()
        for (res_name, res_value) in results['MLP'].items():
            create_metric_expander(results, res_name, res_name == 'Accuracy')

with tab4:
    if st.session_state.X_test is not None:
        patient_id = st.number_input("Enter Patient Id:", 0, st.session_state.X_test.shape[0] - 1, 0, placeholder="Patient Id")
        x = st.session_state.X_test[patient_id]
        y = st.session_state.y_test[patient_id]

        table_contents = pd.DataFrame(
            dict(zip(get_feature_names(), st.session_state.X_test[patient_id]))
            , index=[0]
        ).T.reset_index()
        table_contents.columns = ['Feature', 'Value']

        st.subheader('Feature Values:')
        st.table(table_contents.style.hide(axis='index'))
        st.markdown(f'**True label:** {y.item()}')
        predict = st.button("Predict!")
        if predict:
            predict_and_explain_random(x, y)
    else:
        st.warning("Please upload a dataset to get predictions.")

