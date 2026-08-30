"""Automated Test Suite for Bank Loan Risk Prediction System.

Author: Yordanos Andargachew (Phone: +251 952 190 305)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from src.preprocessor import CreditDataPreprocessor
from src.predictor import LoanRiskPredictor
from cli import app

runner = CliRunner()


@pytest.fixture
def sample_applicant():
    return {
        "Age": 35,
        "Sex": "male",
        "Job": 2,
        "Housing": "own",
        "Saving accounts": "little",
        "Checking account": "moderate",
        "Credit amount": 5000,
        "Duration": 24,
        "Purpose": "car",
    }


def test_dataset_exists_and_valid():
    """Verify dataset.csv exists and has 1000 rows and correct columns."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
    assert os.path.exists(data_path), "dataset.csv does not exist"
    df = pd.read_csv(data_path)
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    expected_cols = [
        "Age", "Sex", "Job", "Housing", "Saving accounts",
        "Checking account", "Credit amount", "Duration", "Purpose", "Risk"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_model_artifacts_exist():
    """Verify preprocessor, logistic regression, and random forest models exist."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    assert os.path.exists(os.path.join(models_dir, "preprocessor.pkl"))
    assert os.path.exists(os.path.join(models_dir, "logistic_regression.pkl"))
    assert os.path.exists(os.path.join(models_dir, "random_forest.pkl"))


def test_preprocessor_transform(sample_applicant):
    """Test preprocessor transforms dictionary input into numerical matrix."""
    preprocessor = CreditDataPreprocessor.load(
        os.path.join(os.path.dirname(__file__), "..", "models", "preprocessor.pkl")
    )
    matrix = preprocessor.transform(sample_applicant)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == 1
    assert matrix.shape[1] == len(preprocessor.get_feature_names_out())


def test_predictor_rf(sample_applicant):
    """Test predictor with Random Forest model."""
    predictor = LoanRiskPredictor()
    res = predictor.predict(sample_applicant, model_name="random_forest")
    assert "risk_class" in res
    assert res["risk_class"] in ["Good Risk", "Bad Risk"]
    assert "prob_good" in res
    assert "prob_bad" in res
    assert 0.0 <= res["prob_good"] <= 1.0
    assert 0.0 <= res["prob_bad"] <= 1.0
    assert round(res["prob_good"] + res["prob_bad"], 2) == 1.0
    assert "confidence_score" in res
    assert "recommendation" in res


def test_predictor_lr(sample_applicant):
    """Test predictor with Logistic Regression model."""
    predictor = LoanRiskPredictor()
    res = predictor.predict(sample_applicant, model_name="logistic_regression")
    assert "risk_class" in res
    assert res["risk_class"] in ["Good Risk", "Bad Risk"]
    assert 0.0 <= res["confidence_score"] <= 100.0


def test_cli_predict():
    """Test CLI predict command via CliRunner."""
    result = runner.invoke(app, [
        "predict",
        "--age", "35",
        "--amount", "5000",
        "--duration", "24",
        "--housing", "own",
        "--saving", "little",
        "--checking", "moderate",
        "--purpose", "car"
    ])
    assert result.exit_code == 0
    assert "Risk Assessment Verdict" in result.output


def test_cli_evaluate():
    """Test CLI evaluate command via CliRunner."""
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 0
    assert "Model Performance Comparison" in result.output
    assert "Logistic" in result.output
    assert "Regression" in result.output
    assert "Random Forest" in result.output
