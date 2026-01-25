"""
Model Service - Handles model training, inference, and evaluation.
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from paths import *


class RulesModel:
    """Model that makes predictions based on rules only."""

    def __init__(self):
        self.name = "RULES"

    def score_samples(self, X, y):
        """
        Score samples using rules.

        Args:
            X: Input features
            y: Target labels

        Returns:
            Prediction scores
        """
        import models.create_rules_only as create_rules_only
        return create_rules_only.predict(X, y)

    def predict(self, X, y):
        """
        Predict labels using rules.

        Args:
            X: Input features
            y: Target labels

        Returns:
            Binary predictions (0 or 1)
        """
        preds = self.score_samples(X, y)
        return preds >= 0.5


class InferenceModel:
    """Model that loads a trained Keras model for inference."""

    def __init__(self, model_name: str):
        """
        Initialize the inference model.

        Args:
            model_name: Name of the model (e.g., 'mlp', 'ltn')
        """
        self.base_model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{model_name}.h5')
        self.name = model_name.upper()

    def score_samples(self, X, y):
        """
        Score samples using the trained model.

        Args:
            X: Input features
            y: Target labels (not used, kept for interface consistency)

        Returns:
            Prediction scores
        """
        return self.base_model(X)

    def predict(self, X, y):
        """
        Predict labels using the trained model.

        Args:
            X: Input features
            y: Target labels (not used, kept for interface consistency)

        Returns:
            Binary predictions (0 or 1)
        """
        preds = self.score_samples(X, y)
        return preds > 0.5


def run_training_pipeline(seed: int, epochs: int) -> None:
    """
    Run the complete training pipeline for both MLP and LTN models.

    Args:
        seed: Random seed for reproducibility
        epochs: Number of training epochs
    """
    # Train LTN model with rules
    os.system(f"python {BASE_DIR}/models/train_ltn.py --input_path={OUTPUT_DIR} --seed={seed} --use_rules=1 --rules_file_path={LTN_RULES_PATH} --epochs={epochs}")

    # Train MLP model without rules
    os.system(f"python {BASE_DIR}/models/train_ltn.py --input_path={OUTPUT_DIR} --seed={seed} --epochs={epochs}")


def run_all_experiments(X_test: np.ndarray, y_test: np.ndarray,
                        seed: int, epochs: int) -> tuple:
    """
    Run all experiments and evaluate all models.

    Args:
        X_test: Test features
        y_test: Test labels
        seed: Random seed for reproducibility
        epochs: Number of training epochs

    Returns:
        Tuple of (results_dict, models_list) where:
        - results_dict: Dictionary with results for each model containing metrics
          (Accuracy, AUROC, F1, Precision, Recall)
        - models_list: List of trained model instances [RulesModel, MLP, LTN]
    """
    results = {}

    # Run training pipeline
    run_training_pipeline(seed, epochs)

    # Initialize models
    models = [
        RulesModel(),
        InferenceModel("mlp"),
        InferenceModel("ltn")
    ]

    # Evaluate each model
    for model in models:
        yp = model.predict(X_test, y_test)
        f1 = f1_score(y_test, yp)
        acc = balanced_accuracy_score(y_test, yp)
        auroc = roc_auc_score(
            y_test,
            model.score_samples(X_test, y_test),
            multi_class='ovr',
            average='weighted'
        )
        report = classification_report(y_test, yp, output_dict=True)
        prec = report[str(1)]['precision']
        recall = report[str(1)]['recall']

        results[model.name] = {
            'Accuracy': acc,
            'AUROC': auroc,
            'F1': f1,
            'Precision': prec,
            'Recall': recall
        }

    return results, models


def predict_sample(model, x: np.ndarray, y: int) -> float:
    """
    Predict a single sample using the given model.

    Args:
        model: Model instance
        x: Input features for a single sample
        y: True label

    Returns:
        Prediction score
    """
    return model.score_samples(np.expand_dims(x, axis=0), [y])[0]
