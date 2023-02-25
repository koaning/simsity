from simsity.service import Service
from simsity.indexer import AnnoyIndexer
from simsity.preprocessing import Identity
from sklearn.datasets import load_iris


def test_smoke():
    """
    Run a simple smoke test to ensure that the service is working.
    """
    # Query an example from the training set
    X, y = load_iris(return_X_y=True)
    service = Service(encoder=Identity(), indexer=AnnoyIndexer(), data=X)
    idx, dists = service.query(
        [5.1, 3.3, 1.7, 0.5],
        n_neighbors=10,
    )

    # The minimum distance should be zero
    assert dists[0] == 0.0
    assert len(idx) == 10
