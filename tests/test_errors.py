import pytest

from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer
from sklearn.feature_extraction.text import CountVectorizer


def test_query_raises_error_no_train():
    """
    You cannot query without training.
    """
    service = Service(
        indexer=PyNNDescentIndexer(metric="euclidean"), encoder=CountVectorizer()
    )
    with pytest.raises(RuntimeError):
        service.query(text="give me directions", n_neighbors=100)


def test_train_save_error(tmpdir):
    """
    You cannot save without training.
    """
    service = Service(
        indexer=PyNNDescentIndexer(metric="euclidean"), encoder=CountVectorizer()
    )
    with pytest.raises(RuntimeError):
        service.save(tmpdir)
