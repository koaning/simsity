from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnGrabber(BaseEstimator, TransformerMixin):
    """Takes a pandas column as a list of text."""

    def __init__(self, column) -> None:
        self.column = column

    def fit(self, X, y) -> "ColumnGrabber":
        """Fits the estimator. No-op."""
        return self

    def transform(self, X) -> List:
        """Transforms the data per scikit-learn API."""
        return X[self.column].to_list()

    def fit_transform(self, X, y=None, **fit_params) -> List:
        """Transforms the data per scikit-learn API."""
        return self.fit(X, y).transform(X)
