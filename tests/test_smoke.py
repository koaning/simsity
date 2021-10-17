from simsity.service import Service


def test_smoke_iris(iris_service, tmpdir):
    """
    Run a simple smoke test to ensure that the service is working.
    """
    # Query an example from the training set
    res = iris_service.query(
        sepal_length=5.1,
        sepal_width=3.3,
        petal_length=1.7,
        petal_width=0.5,
        n_neighbors=10,
    )

    # The minimum distance should be zero
    assert res[0]["dist"] == 0.0
    assert len(res) == 10

    iris_service.save(tmpdir)

    reloaded = Service.load(tmpdir)

    assert reloaded._trained


def test_smoke_clinc(clinc_service, tmpdir):
    """
    Run a simple smoke test to ensure that the service is working.
    """
    res = clinc_service.query(text="hello there", n_neighbors=10)

    assert res[0]["dist"] == 0.0
    assert len(res) == 10

    clinc_service.save(tmpdir)

    reloaded = Service.load(tmpdir)

    assert reloaded._trained
