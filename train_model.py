"""Model training and evaluation pipeline for AgriSmart AI.

This script executes the complete machine learning workflow:
1. Loads and validates the crop recommendation dataset.
2. Checks for missing values, duplicates, and invalid records.
3. Performs a stratified train-test split (80% train, 20% test).
4. Trains a Random Forest Classifier and compares it with a Decision Tree Classifier.
5. Evaluates models strictly on unseen test data to prevent data leakage.
6. Computes accuracy, macro precision, recall, F1-score, and confusion matrix.
7. Saves the trained model artifact (.joblib) and measured evaluation results (.json).
"""

import sys
import os
import json
import pandas as pd
from src.data_loader import (
    load_dataset,
    get_data_summary,
    split_features_target,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.model_utils import (
    split_data,
    train_random_forest,
    train_decision_tree,
    evaluate_model,
    get_feature_importances,
    save_model_artifacts,
)


def main():
    print("=" * 70)
    print("      AgriSmart AI — Model Training & Evaluation Pipeline")
    print("=" * 70)

    # 1. Dataset path
    dataset_path = os.path.join("data", "Crop_recommendation.csv")
    print(f"\n[Step 1] Loading and validating dataset from: {dataset_path}")

    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        sys.exit(1)

    # 2. Dataset summary
    summary = get_data_summary(df)
    print(f" -> Total valid samples: {summary['total_samples']}")
    print(f" -> Input features ({summary['total_features']}): {FEATURE_COLUMNS}")
    print(f" -> Target classes ({summary['total_classes']}): {summary['classes']}")
    print(f" -> Balanced dataset: {summary['is_balanced']}")

    # 3. Feature/Target split
    print("\n[Step 2] Separating features (X) and target crop labels (y)")
    X, y = split_features_target(df)

    # 4. Stratified Train/Test split
    print("\n[Step 3] Splitting dataset: 80% Training, 20% Held-Out Testing (stratified, seed=42)")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f" -> Training samples: {len(X_train)}")
    print(f" -> Testing samples:  {len(X_test)}")

    # 5. Train Random Forest (Baseline & Primary model)
    print("\n[Step 4] Training Random Forest Classifier (100 estimators)...")
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)

    # 6. Train Decision Tree (Comparison benchmark)
    print("[Step 5] Training Decision Tree Classifier for comparison...")
    dt_model = train_decision_tree(X_train, y_train, random_state=42)

    # 7. Evaluate on held-out test data
    print("\n[Step 6] Evaluating models strictly on held-out test data (Zero Data Leakage)...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    dt_metrics = evaluate_model(dt_model, X_test, y_test)

    # Display comparison
    print("\n" + "=" * 65)
    print("                     EVALUATION RESULTS")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Decision Tree':<15} | {'Random Forest':<15}")
    print("-" * 65)
    print(f"{'Accuracy':<25} | {dt_metrics['accuracy'] * 100:.2f}%{' ' * 9} | {rf_metrics['accuracy'] * 100:.2f}%")
    print(f"{'Precision (Macro)':<25} | {dt_metrics['precision_macro'] * 100:.2f}%{' ' * 9} | {rf_metrics['precision_macro'] * 100:.2f}%")
    print(f"{'Recall (Macro)':<25} | {dt_metrics['recall_macro'] * 100:.2f}%{' ' * 9} | {rf_metrics['recall_macro'] * 100:.2f}%")
    print(f"{'F1-Score (Macro)':<25} | {dt_metrics['f1_macro'] * 100:.2f}%{' ' * 9} | {rf_metrics['f1_macro'] * 100:.2f}%")
    print("=" * 65)

    # 8. Feature Importance
    feature_importances = get_feature_importances(rf_model, FEATURE_COLUMNS)
    print("\n[Step 7] Random Forest Feature Importances (Gini importance):")
    for feat, imp in feature_importances.items():
        print(f" -> {feat:<15}: {imp * 100:.2f}%")

    # 9. Save artifacts
    print("\n[Step 8] Saving serialized model and verified metrics...")
    model_output_path = os.path.join("models", "crop_rf_model.joblib")
    metrics_output_path = os.path.join("models", "model_metrics.json")

    # Package comprehensive metadata for the application
    artifacts_data = {
        "model_name": "Random Forest Classifier",
        "n_estimators": 100,
        "random_state": 42,
        "feature_names": FEATURE_COLUMNS,
        "target_classes": summary["classes"],
        "dataset_summary": {
            "total_samples": summary["total_samples"],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_classes": summary["total_classes"],
            "feature_stats": summary["feature_stats"],
            "class_counts": summary["class_counts"],
        },
        "evaluation_metrics": {
            "random_forest": rf_metrics,
            "decision_tree": dt_metrics,
        },
        "feature_importances": feature_importances,
    }

    save_model_artifacts(
        model=rf_model,
        model_path=model_output_path,
        metrics=artifacts_data,
        metrics_path=metrics_output_path,
    )

    print("\n" + "=" * 70)
    print(" Pipeline execution complete! Model ready for deployment.")
    print("=" * 70)


if __name__ == "__main__":
    main()
