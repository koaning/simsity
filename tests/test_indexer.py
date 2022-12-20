import pytest
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

from simsity.indexer import AnnoyIndexer, PyNNDescentIndexer


@pytest.mark.parametrize("indexer", [AnnoyIndexer(), PyNNDescentIndexer()])
def test_basics(indexer):
    """Test the basic API of each indexer"""
    df_clinc = pd.read_csv("tests/data/clinc-data.csv")
    encoder = make_pipeline(CountVectorizer())
    # We're casting to an array to make it non-sparse
    # Not every indexer supports sparse arrays
    X = encoder.fit_transform(df_clinc["text"]).toarray()
    indexer.index(X)
    q = encoder.transform(["translate in english"]).toarray()[0]
    
    print(X)
    print(q)
    
    idx, _ = indexer.query(q, n_neighbors=1)
    assert len(idx) == 1

    idx, _ = indexer.query(q, n_neighbors=10)
    assert len(idx) == 1
