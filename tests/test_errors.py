import json
import pytest
import pathlib

from sklearn.feature_extraction.text import CountVectorizer

from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer


def test_query_raises_error_no_train():
    """
    You cannot query without training.
    """
    service = Service(
        indexer=PyNNDescentIndexer(metric="euclidean"), encoder=CountVectorizer()
    )
    with pytest.raises(RuntimeError):
        service.query(text="give me directions", n_neighbors=100)


def test_train_path_no_exists(tmpdir):
    """
    You cannot load in a folder that does not exist.
    """
    with pytest.raises(FileNotFoundError):
        Service.load(tmpdir)


def test_train_save_error(tmpdir):
    """
    You cannot save without training.
    """
    service = Service(
        encoder=CountVectorizer(),
        indexer=PyNNDescentIndexer(metric="euclidean"),
    )
    with pytest.raises(RuntimeError):
        service.save(tmpdir)


def test_version_load_error(test_smoke_clinc, tmpdir):
    """
    The metadata needs to state the same version.
    """
    test_smoke_clinc.save(tmpdir)
    metadata_file = pathlib.Path(tmpdir) / "metadata.json"
    metadata_file.write_text(json.dumps({"version": "0.0.0"}))
    with pytest.raises(RuntimeError):
        Service.load(tmpdir)
