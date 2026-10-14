"""Unit and integration tests for AgriSmart AI.

Covers:
- Dataset loading and schema validation
- Model artifact availability and metrics schema
- Prediction engine inference and top candidates
- Physiological boundary validation and error handling
"""

import os
import json
import pytest
import pandas as pd
from src.data_loader import (
    load_dataset,
    get_data_summary,
    split_features_target,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
)
from src.predict import PredictionEngine, VALIDATION_CONSTRAINTS, CROP_PROFILES


@pytest.fixture
def dataset_path():
    """Path to the standard dataset."""
    return os.path.join("data", "Crop_recommendation.csv")


@pytest.fixture
def engine():
    """Fixture providing an initialized PredictionEngine instance."""
    return PredictionEngine()


def test_dataset_loading_and_schema(dataset_path):
    """Verify that dataset loads cleanly with exact expected schema and 2200 rows."""
    df = load_dataset(dataset_path)
    assert not df.empty
    assert len(df) == 2200
    for col in EXPECTED_COLUMNS:
        assert col in df.columns
    assert df[FEATURE_COLUMNS].isnull().sum().sum() == 0


def test_dataset_missing_file_raises():
    """Verify that loading a non-existent dataset raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset("data/non_existent_dataset.csv")


def test_data_summary_integrity(dataset_path):
    """Verify data summary returns correct class counts and balanced flag."""
    df = load_dataset(dataset_path)
    summary = get_data_summary(df)
    assert summary["total_samples"] == 2200
    assert summary["total_features"] == 7
    assert summary["total_classes"] == 22
    assert summary["is_balanced"] is True


def test_model_artifacts_exist():
    """Verify that the serialized model and metrics files exist on disk."""
    model_path = os.path.join("models", "crop_rf_model.joblib")
    metrics_path = os.path.join("models", "model_metrics.json")

    assert os.path.exists(model_path), f"Missing model artifact at {model_path}"
    assert os.path.exists(metrics_path), f"Missing metrics artifact at {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "evaluation_metrics" in metrics
    assert "random_forest" in metrics["evaluation_metrics"]
    rf_metrics = metrics["evaluation_metrics"]["random_forest"]

    assert rf_metrics["accuracy"] >= 0.90
    assert rf_metrics["f1_macro"] >= 0.90
    assert len(rf_metrics["classes"]) == 22


def test_prediction_engine_valid_inference(engine):
    """Verify that valid agro-climatic inputs return expected recommendation structure."""
    rice_inputs = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }

    result = engine.predict(rice_inputs, top_k=3)

    assert "predicted_crop" in result
    assert result["predicted_crop"] == "rice"
    assert result["confidence_pct"] > 50.0
    assert len(result["top_candidates"]) == 3
    assert "profile" in result
    assert result["profile"]["display_name"] == "Rice (Oryza sativa)"
    assert "soil_type" in result["profile"]


def test_prediction_engine_alternative_crop(engine):
    """Verify that chickpea/arid inputs predict accurately."""
    chickpea_inputs = {
        "N": 40.0,
        "P": 68.0,
        "K": 79.0,
        "temperature": 18.5,
        "humidity": 17.0,
        "ph": 7.3,
        "rainfall": 75.0,
    }

    result = engine.predict(chickpea_inputs, top_k=3)
    assert result["predicted_crop"] == "chickpea"
    assert result["confidence_pct"] > 50.0


def test_prediction_boundary_validation_fails(engine):
    """Verify that physiological out-of-bound inputs raise ValueError."""
    # Test invalid pH (e.g. 18.0 is chemically invalid for soil)
    invalid_inputs = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 18.5,  # Invalid
        "rainfall": 200.0,
    }

    with pytest.raises(ValueError) as excinfo:
        engine.predict(invalid_inputs)
    assert "Soil pH" in str(excinfo.value)

    # Test negative rainfall
    invalid_rain = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": -45.0,  # Invalid
    }
    with pytest.raises(ValueError) as excinfo:
        engine.predict(invalid_rain)
    assert "Rainfall" in str(excinfo.value)


def test_prediction_missing_features_fails(engine):
    """Verify that omitting required features triggers validation failure."""
    incomplete_inputs = {
        "N": 90.0,
        "P": 42.0,
        # K is missing
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 200.0,
    }
    with pytest.raises(ValueError) as excinfo:
        engine.predict(incomplete_inputs)
    assert "Missing required parameter" in str(excinfo.value)


def test_all_22_crop_profiles_exist():
    """Verify that all 22 target crops have complete curated agronomic profiles."""
    expected_crops = [
        "apple", "banana", "blackgram", "chickpea", "coconut", "coffee",
        "cotton", "grapes", "jute", "kidneybeans", "lentil", "maize",
        "mango", "mothbeans", "mungbean", "muskmelon", "orange", "papaya",
        "pigeonpeas", "pomegranate", "rice", "watermelon",
    ]

    for crop in expected_crops:
        assert crop in CROP_PROFILES, f"Crop profile missing for: {crop}"
        profile = CROP_PROFILES[crop]
        assert "display_name" in profile
        assert "category" in profile
        assert "soil_type" in profile
        assert "water_requirement" in profile
