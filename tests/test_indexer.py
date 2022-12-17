import pytest 
from sklearn.feature_extraction.text import CountVectorizer

from simsity.indexer import AnnoyIndexer, PyNNDescentIndexer


@pytest.mark.parametrize("indexer", [AnnoyIndexer(), PyNNDescentIndexer()])
def test_basics(indexer):
    texts = ["this is text", "bla bla bla"]
    cv = CountVectorizer()
    X = cv.fit_transform(texts)
    indexer.index(X)
    idx, dist = indexer.query(cv.transform("bla")[0], n_neighbors=1)
    assert idx == 1
