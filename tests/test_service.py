from simsity.service import Service
from simsity.indexer import AnnoyIndexer
from simsity.preprocessing import Random
from sklearn.datasets import load_iris


def test_smoke(tmpdir):
    """
    Run a simple smoke test to ensure that the service is working.
    """
    X = [f"this is example {i}" for i in range(200)]
    service = Service(Random(size=100), AnnoyIndexer("euclidean"), X)
    service.index()
    idx, dists = service.query(
        X[0],
        n_neighbors=10,
    )
    # The minimum distance should be zero
    assert len(dists) == len(idx) == 10
    service.to_disk(tmpdir)
    loaded_service = Service.from_disk(tmpdir)
    idx, dists = loaded_service.query(X[0])
    assert len(dists) == len(idx) == 10


if __name__ == "__main__":
    test_smoke()
