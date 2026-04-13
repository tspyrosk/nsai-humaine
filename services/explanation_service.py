"""
Explanation Service - Handles SHAP explanations and XAI operations.
"""
import time

import numpy as np
import shap
from services.data_service import get_feature_names


def get_predicted_class(pred_score: float, target_column: str) -> str:
    """
    Get the predicted class label from a prediction score.

    Args:
        pred_score: Prediction score (0-1)
        target_column: Name of the target column

    Returns:
        String description of the predicted class
    """
    if pred_score > 0.5:
        cl = target_column.lower()
    else:
        cl = f"no {target_column.lower()}"
    return cl


def explain_predictions(model, x: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """
    Generate SHAP explanations for predictions.

    Args:
        model: Model instance with score_samples method
        x: Input sample(s) to explain
        X_train: Training data to use as background for SHAP

    Returns:
        SHAP values for the input sample(s)
    """
    def f(x):
        return model.score_samples(x, np.zeros(x.shape[0]))

    background = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(x, nsamples=100)
    return shap_values


def top_3_shap_features(shap_values: np.ndarray) -> list:
    """
    Get the top 3 most important features based on SHAP values.

    Args:
        shap_values: SHAP values array

    Returns:
        List of top 3 feature names
    """
    feature_names = get_feature_names()
    top_3_idx = np.argsort(shap_values, axis=0)[-3:][::-1].tolist()
    return [feature_names[idx[0]] for idx in top_3_idx]


def get_rag_explanation(concepts: list, important_features: list,
                        predicted_class: str, dataset_description: dict = None) -> str:
    """
    Generate RAG-based explanation for the prediction.

    Args:
        concepts: List of satisfied concepts/rules
        important_features: List of important feature names
        predicted_class: Predicted class label
        dataset_description: Optional dict with domain, row_description, prediction_target, class_descriptions

    Returns:
        Natural language explanation string
    """
    import models.xai_rag as xai_rag
    return xai_rag.extract_rag_explanation(concepts, important_features, predicted_class, dataset_description)


def get_satisfied_rules(x: np.ndarray, y: int, selected_rules: list) -> list:
    """
    Get the rules that are satisfied for a given sample.

    Args:
        x: Input sample
        y: True label
        selected_rules: List of all available rules

    Returns:
        List of satisfied rule descriptions
    """
    import models.create_rules_only as create_rules_only
    rules_triggered = create_rules_only.get_satisfied_rule_indexes(x, y)

    satisfied = []
    if len(rules_triggered) > 0 and len(selected_rules) > 0:
        for rule_idx in rules_triggered:
            satisfied.append(selected_rules[rule_idx])

    return satisfied


def get_satisfied_concepts(x: np.ndarray) -> list:
    """
    Get the concepts that are satisfied for a given sample.

    Args:
        x: Input sample

    Returns:
        List of satisfied concepts
    """
    import models.create_rules_only as create_rules_only
    return create_rules_only.satisfied_concepts(x)


def predict_and_explain(x: np.ndarray, y: int, X_train: np.ndarray,
                        target_column: str, selected_rules: list,
                        dataset_description: dict = None) -> dict:
    """
    Make predictions with all models and generate explanations.

    Args:
        x: Input sample to predict
        y: True label
        X_train: Training data for SHAP background
        target_column: Name of the target column
        selected_rules: List of all available rules
        dataset_description: Optional dict with domain, row_description, prediction_target, class_descriptions

    Returns:
        Dictionary containing:
        - scores: Prediction scores for each model
        - shap_values: SHAP values for LTN model
        - satisfied_rules: List of satisfied rule descriptions
        - rag_explanation: Natural language explanation
        - important_features: Top 3 important features
    """
    from services.model_service import RulesModel, InferenceModel, predict_sample

    # Get trained models (loads whatever .h5 files exist)
    trained_models = [
        RulesModel(),
        InferenceModel("mlp"),
        InferenceModel("ltn")
    ]

    # Collect predictions and SHAP values
    scores = {}
    collected_shap_values = []

    t0 = time.time()
    for trained_model in trained_models:
        pred = predict_sample(trained_model, x, y)
        scores[trained_model.name] = pred
        shap_values = explain_predictions(trained_model, np.expand_dims(x, axis=0), X_train)
        collected_shap_values.append(shap_values[0])
    prediction_latency_ms = round((time.time() - t0) * 1000, 1)

    # Get rule and concept explanations
    satisfied_rules = get_satisfied_rules(x, y, selected_rules)
    concepts = get_satisfied_concepts(x)
    important_features = top_3_shap_features(collected_shap_values[2])

    # Generate RAG explanation using LTN model score
    ltn_score = scores.get("LTN", 0.5)
    t1 = time.time()
    rag_explanation = get_rag_explanation(
        concepts,
        important_features,
        get_predicted_class(ltn_score, target_column),
        dataset_description
    )
    rag_latency_ms = round((time.time() - t1) * 1000, 1)

    return {
        'scores': scores,
        'shap_values': collected_shap_values[2].flatten(),
        'satisfied_rules': satisfied_rules,
        'rag_explanation': rag_explanation,
        'important_features': important_features,
        'prediction_latency_ms': prediction_latency_ms,
        'rag_latency_ms': rag_latency_ms,
    }
