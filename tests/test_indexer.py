import pytest
import numpy as np

from simsity.indexer import AnnoyIndexer, NMSlibIndexer


@pytest.mark.parametrize("indexer", [AnnoyIndexer, NMSlibIndexer])
def test_can_index_retreive_save_load(indexer, tmpdir):
    """Test retreival basics"""
    indexer_obj = indexer()
    data = np.random.normal(0, 1, (100, 10))
    indexer_obj.index(data)
    idx, dist = indexer_obj.query(data[0])
    assert idx[0] == 0
    assert np.isclose(dist[0], 0.0, atol=0.0001)
    indexer_obj.save(tmpdir)
    loader_indexer = indexer.load(tmpdir)
    idx, dist = loader_indexer.query(data[0])
    assert idx[0] == 0
    assert np.isclose(dist[0], 0.0, atol=0.0001)


@pytest.mark.parametrize("indexer", [AnnoyIndexer, NMSlibIndexer])
def test_throw_error_not_indexed(indexer):
    """If we haven't indexer yet: it should err"""
    indexer_obj = indexer()
    data = np.random.normal(0, 1, (100, 10))
    with pytest.raises(RuntimeError):
        indexer_obj.query(data[0])
