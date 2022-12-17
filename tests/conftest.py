import pytest

import pandas as pd

from simsity.service import Service
from simsity.indexer import AnnoyIndexer, PyNNDescentIndexer
from simsity.preprocessing import Identity

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer


@pytest.fixture(scope="session")
def iris_service():
    """Create a service trained on iris."""
    X, y = load_iris(return_X_y=True)
    service_iris = Service(
        encoder=Identity(),
        indexer=AnnoyIndexer(metric="euclidean", n_trees=10),
    )
    service_iris.index(X)

    return service_iris


@pytest.fixture(scope="session")
def clinc_service():
    """Create a service trained on clinc."""
    df_clinc = pd.read_csv("tests/data/clinc-data.csv")
    encoder = make_pipeline(CountVectorizer())
    encoder.fit(df_clinc["text"])
    indexer = PyNNDescentIndexer(metric="euclidean", n_neighbors=2)
    service_clinc = Service(
        encoder=encoder,
        indexer=indexer,
    )
    service_clinc.index(df_clinc["text"])
    return service_clinc
