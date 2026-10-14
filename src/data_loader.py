"""Data loader and validation module for AgriSmart AI.

This module provides functions to safely load, inspect, and validate the
Crop Recommendation dataset, ensuring schema conformity, checking physiological
ranges, and detecting data quality issues.
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

# Expected schema for the crop recommendation dataset
EXPECTED_COLUMNS: List[str] = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "label",
]

FEATURE_COLUMNS: List[str] = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

TARGET_COLUMN: str = "label"

# Realistic agronomic boundaries for validation
FEATURE_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "N": (0.0, 300.0, "kg/ha"),
    "P": (0.0, 300.0, "kg/ha"),
    "K": (0.0, 300.0, "kg/ha"),
    "temperature": (-10.0, 60.0, "°C"),
    "humidity": (0.0, 100.0, "%"),
    "ph": (0.0, 14.0, "pH"),
    "rainfall": (0.0, 4000.0, "mm"),
}


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the dataset from CSV and perform comprehensive validation.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned and validated dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If columns are missing or data types are invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please ensure Crop_recommendation.csv is located in the data/ folder."
        )

    # Read CSV
    df = pd.read_csv(filepath)

    # Strip any accidental whitespace from column names
    df.columns = [col.strip() for col in df.columns]

    # Validate schema
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset schema mismatch. Missing required columns: {missing_cols}. "
            f"Expected schema: {EXPECTED_COLUMNS}"
        )

    # Check for empty dataframe
    if df.empty:
        raise ValueError("The provided dataset is empty (0 rows).")

    # Check for missing (NaN / null) values
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        print(f"[Warning] Found {total_nulls} missing values across columns:")
        for col, cnt in null_counts.items():
            if cnt > 0:
                print(f"  - {col}: {cnt} missing")
        # Remove incomplete records to maintain data integrity
        df = df.dropna(subset=EXPECTED_COLUMNS).copy()
        print(f"[Info] Remaining valid records after dropna: {len(df)}")

    # Check for duplicate rows
    duplicate_count = int(df.duplicated(subset=EXPECTED_COLUMNS).sum())
    if duplicate_count > 0:
        print(f"[Warning] Detected {duplicate_count} duplicate rows. Removing duplicates.")
        df = df.drop_duplicates(subset=EXPECTED_COLUMNS).copy()

    # Enforce numeric types for features
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            raise ValueError(
                f"Feature '{col}' contains non-numeric or invalid values."
            )

    # Ensure label column is clean string format
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower()

    # Check physiological bounds
    for col, (min_val, max_val, unit) in FEATURE_BOUNDS.items():
        out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
        if not out_of_bounds.empty:
            print(
                f"[Warning] {len(out_of_bounds)} records in '{col}' fall outside "
                f"expected range [{min_val}, {max_val} {unit}]."
            )

    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a statistical and structural summary of the dataset.

    Args:
        df: Validated DataFrame.

    Returns:
        Dict[str, Any]: Summary dictionary containing statistics and class distributions.
    """
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    feature_stats = df[FEATURE_COLUMNS].describe().to_dict()

    return {
        "total_samples": int(len(df)),
        "total_features": len(FEATURE_COLUMNS),
        "total_classes": len(class_counts),
        "classes": sorted(list(class_counts.keys())),
        "class_counts": class_counts,
        "is_balanced": (
            max(class_counts.values()) - min(class_counts.values()) <= 5
            if class_counts
            else True
        ),
        "feature_stats": feature_stats,
    }


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features from target crop label.

    Args:
        df: Validated DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and Target (y).
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
