"""Risk scoring engine and probability calibrator for Bank Loan Risk Prediction.

Author: Yordanos Andargachew (Phone: +251 952 190 305)
"""

from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .preprocessor import CreditDataPreprocessor


class LoanRiskPredictor:
    """Production inference engine for scoring loan applicant default risk."""

    MODEL_ALIASES: Dict[str, str] = {
        "rf": "random_forest",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "lr": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "logistic": "logistic_regression",
    }

    def __init__(
        self,
        models_dir: Optional[str] = None,
        preprocessor: Optional[CreditDataPreprocessor] = None,
        rf_model: Optional[Any] = None,
        lr_model: Optional[Any] = None,
    ) -> None:
        """Initialize the LoanRiskPredictor by loading serialized models or using passed instances."""
        if models_dir is None:
            # Default to ../models relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        self.models_dir = models_dir

        self.preprocessor: Optional[CreditDataPreprocessor] = preprocessor
        self.models: Dict[str, Any] = {}

        if rf_model is not None:
            self.models["random_forest"] = rf_model
        if lr_model is not None:
            self.models["logistic_regression"] = lr_model

        # Attempt to load from disk if not explicitly passed
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load preprocessor and models from the models directory if present."""
        if self.preprocessor is None:
            prep_path = os.path.join(self.models_dir, "preprocessor.pkl")
            if os.path.exists(prep_path):
                self.preprocessor = CreditDataPreprocessor.load(prep_path)

        if "random_forest" not in self.models:
            rf_path = os.path.join(self.models_dir, "random_forest.pkl")
            if os.path.exists(rf_path):
                self.models["random_forest"] = joblib.load(rf_path)

        if "logistic_regression" not in self.models:
            lr_path = os.path.join(self.models_dir, "logistic_regression.pkl")
            if os.path.exists(lr_path):
                self.models["logistic_regression"] = joblib.load(lr_path)

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model alias to canonical key."""
        key = model_name.strip().lower()
        if key not in self.MODEL_ALIASES:
            valid_keys = list(self.MODEL_ALIASES.keys())
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {valid_keys}")
        canonical = self.MODEL_ALIASES[key]
        if canonical not in self.models:
            self._load_artifacts()
            if canonical not in self.models:
                raise RuntimeError(
                    f"Model '{canonical}' is not loaded. Ensure '{canonical}.pkl' exists in '{self.models_dir}'."
                )
        return canonical

    def predict(
        self,
        applicant_data: Union[Dict[str, Any], pd.DataFrame],
        model_name: str = "random_forest",
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Assess credit risk for a single applicant or batch.

        Parameters
        ----------
        applicant_data : dict or DataFrame
            Must contain applicant features (Age, Sex, Job, Housing, Saving accounts,
            Checking account, Credit amount, Duration, Purpose).
        model_name : str
            'random_forest' (default) or 'logistic_regression'
        threshold : float
            Default decision boundary probability (default 0.50)

        Returns
        -------
        dict
            Structured assessment containing risk classification, probability breakdown,
            confidence score, risk category tier, and decision recommendation.
        """
        canonical_model_name = self._resolve_model_name(model_name)
        model = self.models[canonical_model_name]

        if self.preprocessor is None:
            self._load_artifacts()
            if self.preprocessor is None:
                raise RuntimeError(
                    f"Preprocessor not found. Please train models or check '{self.models_dir}/preprocessor.pkl'."
                )

        if isinstance(applicant_data, dict):
            input_df = pd.DataFrame([applicant_data])
            is_single = True
        else:
            input_df = applicant_data.copy()
            is_single = len(input_df) == 1

        # Transform features
        X_trans = self.preprocessor.transform(input_df)

        # Probabilities: class 0 is 'Good', class 1 is 'Bad'
        probabilities = model.predict_proba(X_trans)
        prob_good = float(probabilities[0][0])
        prob_bad = float(probabilities[0][1])

        # Prediction decision based on threshold for Bad Risk
        is_bad = prob_bad >= threshold
        risk_class = "Bad Risk" if is_bad else "Good Risk"
        confidence = float(max(prob_good, prob_bad) * 100.0)

        # Qualitative Risk Tier
        if prob_bad < 0.25:
            risk_tier = "LOW RISK"
            recommendation = "APPROVE LOAN: Excellent credit profile and high repayment confidence."
            badge_color = "#10b981"  # Green
        elif prob_bad < 0.50:
            risk_tier = "MODERATE RISK"
            recommendation = "APPROVE WITH STANDARD CONDITIONS: Acceptable risk profile."
            badge_color = "#3b82f6"  # Blue
        elif prob_bad < 0.70:
            risk_tier = "HIGH RISK"
            recommendation = "CAUTION / MANUAL REVIEW: Elevated probability of default. Consider collateral."
            badge_color = "#f59e0b"  # Amber
        else:
            risk_tier = "CRITICAL RISK"
            recommendation = "REJECT LOAN: Severe default risk exceeding acceptable tolerance."
            badge_color = "#ef4444"  # Red

        result = {
            "risk_class": risk_class,
            "risk_tier": risk_tier,
            "is_default_risk": bool(is_bad),
            "prob_good": round(prob_good, 4),
            "prob_bad": round(prob_bad, 4),
            "good_percentage": round(prob_good * 100.0, 2),
            "bad_percentage": round(prob_bad * 100.0, 2),
            "confidence_score": round(confidence, 2),
            "recommendation": recommendation,
            "badge_color": badge_color,
            "model_used": "Random Forest (Advanced Ensemble)" if canonical_model_name == "random_forest" else "Logistic Regression (Linear Baseline)",
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_features": applicant_data if isinstance(applicant_data, dict) else applicant_data.iloc[0].to_dict(),
        }
        return result

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute performance metrics for both models on a test set."""
        self._load_artifacts()
        metrics: Dict[str, Dict[str, Any]] = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            cm = confusion_matrix(y_test, y_pred)
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            roc_auc = float(roc_auc_score(y_test, y_proba))

            metrics[name] = {
                "model_name": "Random Forest" if name == "random_forest" else "Logistic Regression",
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "confusion_matrix": cm.tolist(),
            }

        return metrics
