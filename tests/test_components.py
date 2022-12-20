import pytest
import pandas as pd

from simsity.preprocessing import KeyGrabber, ColumnGrabber, Identity


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
    assert ColumnGrabber(column=colname).fit_transform(dataf) == ["example"]


def test_key_lister():
    """Test the ColumnLister estimator."""
    data = [{"text": "yes"}]
    assert KeyGrabber(key="text").fit_transform(data) == ["yes"]
