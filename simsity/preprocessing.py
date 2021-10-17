from sklearn.base import BaseEstimator, TransformerMixin
from datasketch import MinHash


class Identity(BaseEstimator, TransformerMixin):
    """Encoder/Transformer that keeps data as-is."""

    def __init__(self):
        pass

    def fit(self, X, y):
        """Fits the estimator. No-op."""
        return self

    def transform(self, X):
        """Transforms the data per scikit-learn API."""
        return X


class ColumnLister(BaseEstimator, TransformerMixin):
    """Takes a pandas column as a list of text."""

    def __init__(self, column):
        self.column = column

    def fit(self, X, y):
        """Fits the estimator. No-op."""
        return self

    def transform(self, X):
        """Transforms the data per scikit-learn API."""
        return X[self.column].to_list()


class SparseMinHasher(BaseEstimator, TransformerMixin):
    """
    Turns sparse matrix into a list of minhash objects

    Arguments:
        num_perm: number of permutations to use in the minhash
    """

    def __init__(self, num_perm=128):
        self.num_perm = num_perm

    def to_minhash(self, things):
        """
        Turns a "thing" into a MinHash.
        """
        m = MinHash(num_perm=self.num_perm)
        for thing in things:
            m.update(thing.encode("utf-8"))
        return m

    def fit(self, X, y):
        """Fits the estimator. No-op."""
        return self

    def transform(self, X):
        """Transforms the data per scikit-learn API."""
        return [self.to_minhash({str(idx) for idx in x.indices}) for x in X]
