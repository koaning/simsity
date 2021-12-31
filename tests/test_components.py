import pytest
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from simsity.preprocessing import SparseMinHasher, ColumnLister, Identity


@pytest.mark.parametrize("data", [[1], [1, 2], [1, 2, 3]])
def test_identity(data):
    """Test the Identity estimator."""
    assert Identity().fit_transform(data) == data


@pytest.mark.parametrize(
    "dataf",
    [
        pd.DataFrame({"hello": ["example"]}),
        pd.DataFrame({"text": ["example"]}),
    ],
)
def test_column_lister(dataf):
    """Test the ColumnLister estimator."""
    colname = dataf.columns[0]
    assert ColumnLister(column=colname).fit_transform(dataf) == ["example"]


@pytest.mark.parametrize(
    "data", [["this is text", "so is this"], ["this is text", "so is this", "and this"]]
)
def test_column_min_hash(data):
    """Test the SparseMinHasher estimator."""
    pipe = make_pipeline(CountVectorizer(), SparseMinHasher())
    X_out = pipe.fit_transform(data)
    assert len(X_out) == len(data)
