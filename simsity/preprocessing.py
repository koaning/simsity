from typing import Any, List, Union, Set
from sklearn.base import BaseEstimator, TransformerMixin
from datasketch import MinHash


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


class ColumnLister(BaseEstimator, TransformerMixin):
    """Takes a pandas column as a list of text."""

    def __init__(self, column) -> None:
        self.column = column

    def fit(self, X, y) -> "ColumnLister":
        """Fits the estimator. No-op."""
        return self

    def transform(self, X) -> List[str]:
        """Transforms the data per scikit-learn API."""
        return X[self.column].to_list()

    def fit_transform(self, X, y=None, **fit_params) -> List[str]:
        """Transforms the data per scikit-learn API."""
        return self.fit(X, y).transform(X)


class SparseMinHasher(BaseEstimator, TransformerMixin):
    """
    Turns sparse matrix into a list of minhash objects

    Arguments:
        num_perm: number of permutations to use in the minhash
    """

    def __init__(self, num_perm=128) -> None:
        self.num_perm = num_perm

    def to_minhash(self, things: Union[List[str], Set[str]]) -> MinHash:
        """
        Turns a "thing" into a MinHash.
        """
        m = MinHash(num_perm=self.num_perm)
        for thing in things:
            m.update(thing.encode("utf-8"))
        return m

    def fit(self, X, y) -> "SparseMinHasher":
        """Fits the estimator. No-op."""
        return self

    def transform(self, X) -> List[MinHash]:
        """Transforms the data per scikit-learn API."""
        return [self.to_minhash({str(idx) for idx in x.indices}) for x in X]

    def fit_transform(self, X, y=None, **fit_params) -> List[MinHash]:
        """Transforms the data per scikit-learn API."""
        return self.fit(X, y).transform(X)
