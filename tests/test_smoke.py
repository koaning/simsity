def test_smoke_iris(iris_service):
    """
    Run a simple smoke test to ensure that the service is working.
    """
    # Query an example from the training set
    idx, dists = iris_service.query(
        [5.1, 3.3, 1.7, 0.5],
        n_neighbors=10,
    )

    # The minimum distance should be zero
    assert dists[0] == 0.0
    assert len(idx) == 10


def test_smoke_clinc(clinc_service):
    """
    Run a simple smoke test to ensure that the service is working.
    """
    idx, dists = clinc_service.query("hello there", n_neighbors=10)

    assert dists[0] == 0.0
    assert len(idx) == 10
