"""Data cleaning, encoding, and scaling transformer for Credit Risk Prediction.

Author: Yordanos Andargachew (Phone: +251 952 190 305)
"""

from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CreditDataPreprocessor(BaseEstimator, TransformerMixin):
    """Production preprocessing pipeline for German Credit Risk dataset."""

    NUMERICAL_FEATURES: List[str] = ["Age", "Credit amount", "Duration"]
    CATEGORICAL_FEATURES: List[str] = [
        "Sex",
        "Housing",
        "Saving accounts",
        "Checking account",
        "Purpose",
    ]
    ORDINAL_FEATURES: List[str] = ["Job"]
    ALL_FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    TARGET_COLUMN: str = "Risk"

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def _build_pipeline(self) -> ColumnTransformer:
        """Construct the sklearn ColumnTransformer."""
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop=None,
                    ),
                ),
            ]
        )

        ord_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.NUMERICAL_FEATURES),
                ("cat", cat_pipeline, self.CATEGORICAL_FEATURES),
                ("ord", ord_pipeline, self.ORDINAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return transformer

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "CreditDataPreprocessor":
        """Fit the preprocessor on the training dataframe."""
        df = X.copy()
        # Verify required columns are present
        missing_cols = [col for col in self.ALL_FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input DataFrame: {missing_cols}")

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df[self.ALL_FEATURE_COLUMNS])

        # Extract output feature names
        try:
            self.feature_names_out_ = list(self.pipeline.get_feature_names_out())
        except Exception:
            # Fallback feature naming if older sklearn
            num_names = self.NUMERICAL_FEATURES
            cat_names = list(
                self.pipeline.named_transformers_["cat"]
                .named_steps["encoder"]
                .get_feature_names_out(self.CATEGORICAL_FEATURES)
            )
            ord_names = self.ORDINAL_FEATURES
            self.feature_names_out_ = num_names + cat_names + ord_names

        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """Transform input data into normalized numerical matrix."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() or load() first.")

        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Unsupported input type for transform: {type(X)}")

        # Ensure all columns exist, filling missing categorical columns with 'unknown' if absent
        for col in self.NUMERICAL_FEATURES:
            if col not in df.columns:
                raise ValueError(f"Missing numerical feature: {col}")
        for col in self.CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "unknown"
        if "Job" not in df.columns:
            df["Job"] = 2  # Default to skilled (job code 2)

        return self.pipeline.transform(df[self.ALL_FEATURE_COLUMNS])

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return list of transformed feature names."""
        if not self.is_fitted or self.feature_names_out_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.feature_names_out_

    def save(self, filepath: str) -> None:
        """Serialize fitted preprocessor to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CreditDataPreprocessor":
        """Load serialized preprocessor from disk."""
        instance = joblib.load(filepath)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return instance

    @staticmethod
    def encode_target(y: Union[pd.Series, np.ndarray, List[str]]) -> np.ndarray:
        """Encode target risk: 'good' -> 0 (Low Risk), 'bad' -> 1 (High Risk / Default)."""
        series = pd.Series(y).astype(str).str.strip().str.lower()
        return (series == "bad").astype(int).to_numpy()

    @staticmethod
    def decode_target(y_encoded: Union[np.ndarray, List[int]]) -> List[str]:
        """Decode binary target back to risk labels: 0 -> 'Good Risk', 1 -> 'Bad Risk'."""
        arr = np.array(y_encoded)
        return ["Bad Risk" if val == 1 else "Good Risk" for val in arr]
