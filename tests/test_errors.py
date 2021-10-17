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


def test_version_load_error(clinc_service, tmpdir):
    """
    The metadata needs to state the same version.
    """
    clinc_service.save(tmpdir)
    metadata_file = pathlib.Path(tmpdir) / "metadata.json"
    metadata_file.write_text(json.dumps({"version": "0.0.0"}))
    with pytest.raises(RuntimeError):
        Service.load(tmpdir)


def test_query_smaller_than_data_error(clinc_service):
    """
    You cannot query more than we have in storage.
    """
    with pytest.raises(ValueError):
        clinc_service.query(text="give me directions", n_neighbors=100_000)
