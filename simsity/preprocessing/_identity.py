import numpy as np
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin


class Identity(BaseEstimator, TransformerMixin):
    """Encoder/Transformer that keeps data as-is."""

    def __init__(self) -> None:
        pass

    def fit(self, X, y) -> "Identity":
        """Fits the estimator. No-op."""
        return self

    def transform(self, X: Any) -> Any:
        """Transforms the data per scikit-learn API."""
        return X

    def fit_transform(self, X, y=None, **fit_params) -> Any:
        """Transforms the data per scikit-learn API."""
        return self.fit(X, y).transform(X)


class Random(BaseEstimator, TransformerMixin):
    """Encoder/Transformer that keeps data as-is."""

    def __init__(self, size) -> None:
        self.size = size

    def fit(self, X, y) -> "Identity":
        """Fits the estimator. No-op."""
        return self

    def transform(self, X: Any) -> Any:
        """Transforms the data per scikit-learn API."""
        return np.random.normal(0, 1, (len(X), self.size))

    def fit_transform(self, X, y=None, **fit_params) -> Any:
        """Transforms the data per scikit-learn API."""
        return self.fit(X, y).transform(X)
