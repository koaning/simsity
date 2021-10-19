import pytest

import pandas as pd

from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer, MinHashIndexer
from simsity.preprocessing import Identity, ColumnLister, SparseMinHasher

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer


@pytest.fixture(scope="session")
def iris_service():
    """Create a service trained on iris."""
    df_iris = load_iris(as_frame=True)["data"]
    df_iris.columns = [
        c.replace(" (cm)", "").replace(" ", "_") for c in df_iris.columns
    ]
    service_iris = Service(
        encoder=Identity(),
        indexer=PyNNDescentIndexer(metric="euclidean", n_neighbors=2),
    )
    service_iris.train_from_dataf(df_iris)

    return service_iris


@pytest.fixture(scope="session")
def clinc_service():
    """Create a service trained on clinc."""
    df_clinc = pd.read_csv("tests/data/clinc-data.csv")
    service_clinc = Service(
        encoder=make_pipeline(ColumnLister(column="text"), CountVectorizer()),
        indexer=PyNNDescentIndexer(metric="euclidean", n_neighbors=2),
    )
    service_clinc.train_from_dataf(df_clinc, features=["text"])
    return service_clinc


@pytest.fixture(scope="session")
def clinc_minhash_service():
    """Create a service trained on clinc."""
    df_clinc = pd.read_csv("tests/data/clinc-data.csv")
    service_clinc = Service(
        indexer=MinHashIndexer(),
        encoder=make_pipeline(CountVectorizer(), SparseMinHasher()),
    )
    service_clinc.train_from_dataf(df_clinc, features=["text"])
    return service_clinc


@pytest.fixture(scope="session")
def untrained_service():
    """Create a service that's not seen any data."""
    return Service(
        encoder=make_pipeline(ColumnLister(column="text"), CountVectorizer()),
        indexer=PyNNDescentIndexer(metric="euclidean", n_neighbors=2),
    )
