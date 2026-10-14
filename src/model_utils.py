"""Model training, evaluation, comparison, and serialization utilities for AgriSmart AI.

This module ensures rigorous evaluation, reproducible train/test splits,
prevention of data leakage, and honest metric calculation for both Random Forest
and Decision Tree classifiers.
"""

import json
import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split.

    Stratification ensures every crop class is proportionately represented
    in both the training and test splits, preventing evaluation bias.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of dataset to hold out for testing (default 0.2 = 20%).
        random_state: Fixed seed for exact reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest Classifier.

    Args:
        X_train: Training features.
        y_train: Training targets.
        n_estimators: Number of trees in the forest.
        random_state: Fixed seed for reproducibility.

    Returns:
        Trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=None,
        min_samples_split=2,
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train a single Decision Tree Classifier for baseline comparison.

    Args:
        X_train: Training features.
        y_train: Training targets.
        random_state: Fixed seed for reproducibility.

    Returns:
        Trained DecisionTreeClassifier model.
    """
    model = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=None,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a trained model strictly on the held-out test split.

    Calculates:
    - Overall Accuracy
    - Macro-averaged and Weighted Precision, Recall, and F1-score
    - Confusion Matrix
    - Per-class classification report

    Args:
        model: Trained scikit-learn estimator.
        X_test: Test features.
        y_test: Ground truth test labels.

    Returns:
        Dict[str, Any]: Dictionary of measured evaluation metrics.
    """
    y_pred = model.predict(X_test)
    labels = sorted(list(np.unique(y_test)))

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "classes": labels,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "test_sample_count": int(len(y_test)),
    }


def get_feature_importances(
    model: RandomForestClassifier,
    feature_names: list,
) -> Dict[str, float]:
    """Extract and sort Gini-based feature importances from a Random Forest.

    Args:
        model: Trained RandomForestClassifier.
        feature_names: List of input feature names.

    Returns:
        Dict[str, float]: Feature names mapped to their importance weights.
    """
    importances = model.feature_importances_
    sorted_pairs = sorted(
        zip(feature_names, [round(float(v), 4) for v in importances]),
        key=lambda x: x[1],
        reverse=True,
    )
    return {k: v for k, v in sorted_pairs}


def save_model_artifacts(
    model: Any,
    model_path: str,
    metrics: Dict[str, Any],
    metrics_path: str,
) -> None:
    """Serialize the trained model using joblib and write metrics to JSON.

    Args:
        model: Trained scikit-learn model object.
        model_path: Destination path for .joblib file.
        metrics: Dictionary of evaluated metrics.
        metrics_path: Destination path for .json metrics file.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Save serialized model
    joblib.dump(model, model_path)
    print(f"[Info] Model successfully saved to '{model_path}'.")

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Info] Metrics successfully saved to '{metrics_path}'.")
