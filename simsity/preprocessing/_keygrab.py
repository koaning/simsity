from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class KeyGrabber(BaseEstimator, TransformerMixin):
    """Takes a pandas column as a list of text."""

    def __init__(self, key) -> None:
        self.key = key

    def fit(self, X, y) -> "KeyGrabber":
        """Fits the estimator. No-op."""
        return self

    def transform(self, X) -> List:
        """Transforms the data per scikit-learn API."""
        return [x[self.key] for x in X]

    def fit_transform(self, X, y=None, **fit_params) -> List:
        """Transforms the data per scikit-learn API."""
        return self.fit(X, y).transform(X)
