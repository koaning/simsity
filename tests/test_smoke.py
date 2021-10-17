import pandas as pd

from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer
from simsity.preprocessing import Identity, ColumnLister
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

df_iris = load_iris(as_frame=True)["data"]
df_iris.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df_iris.columns]

df_clinc = pd.read_csv("tests/data/clinc-data.csv")


def test_smoke_iris(tmpdir):
    """
    Run a simple smoke test to ensure that the service is working.
    """
    service = Service(
        encoder=Identity(),
        indexer=PyNNDescentIndexer(metric="euclidean", n_neighbors=2),
    )

    service.train_from_dataf(df_iris)

    # Query an example from the training set
    res = service.query(
        sepal_length=5.1,
        sepal_width=3.3,
        petal_length=1.7,
        petal_width=0.5,
        n_neighbors=10,
    )

    # The minimum distance should be zero
    assert res[0]["dist"] == 0.0
    assert len(res) == 10

    service.save(tmpdir)

    reloaded = Service.load(tmpdir)

    assert reloaded._trained


def test_smoke_clinc(tmpdir):
    """
    Run a simple smoke test to ensure that the service is working.
    """
    service = Service(
        encoder=make_pipeline(ColumnLister(column="text"), CountVectorizer()),
        indexer=PyNNDescentIndexer(metric="euclidean", n_neighbors=2),
    )

    service.train_from_dataf(df_clinc, features=["text"])

    # Query an example from the training set
    res = service.query(text="hello there", n_neighbors=10)
    # The minimum distance should be zero
    assert res[0]["dist"] == 0.0
    assert len(res) == 10

    service.save(tmpdir)

    reloaded = Service.load(tmpdir)

    assert reloaded._trained
