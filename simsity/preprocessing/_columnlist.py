from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnLister(BaseEstimator, TransformerMixin):
    """Takes a pandas column as a list of text."""

    def __init__(self, column) -> None:
        self.column = column
        self._is_fitted = False

    def fit(self, X, y) -> "ColumnLister":
        """Fits the estimator. No-op."""
        self._is_fitted = True
        return self

    def transform(self, X) -> List[str]:
        """Transforms the data per scikit-learn API."""
        return X[self.column].to_list()

    def fit_transform(self, X, y=None, **fit_params) -> List[str]:
        """Transforms the data per scikit-learn API."""
        return self.fit(X, y).transform(X)
