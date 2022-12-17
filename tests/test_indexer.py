import pytest
from sklearn.feature_extraction.text import CountVectorizer

from simsity.indexer import AnnoyIndexer, PyNNDescentIndexer


@pytest.mark.parametrize("indexer", [AnnoyIndexer(), PyNNDescentIndexer()])
def test_basics(indexer):
    """Test the basic API of each indexer"""
    texts = ["this is text", "bla bla bla"]
    cv = CountVectorizer()
    X = cv.fit_transform(texts).todense()
    indexer.index(X)
    q = cv.transform("bla").toarray()[0]
    idx, dist = indexer.query(q, n_neighbors=1)
    assert idx == 1
