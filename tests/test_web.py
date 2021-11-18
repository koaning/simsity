import pytest

from simsity.serve import create_app, create_multi_app
from starlette.testclient import TestClient


@pytest.mark.parametrize(
    "payload, exp",
    [
        ({"query": {"text": "hello there"}, "n_neighbors": 20}, "hello"),
        ({"query": {"text": "hello there"}, "n_neighbors": 10}, "hello"),
        ({"query": {"text": "thanks"}, "n_neighbors": 10}, "thanks"),
    ],
)
def test_basic_post_rqs(clinc_service, payload, exp):
    """Confirm that the basic queries come back right"""
    client = TestClient(create_app(clinc_service))
    resp = client.post("/query", json=payload)
    print(resp.json())
    assert resp.status_code == 200
    assert len(resp.json()) == payload["n_neighbors"]
    assert any(exp in s["item"]["text"] for s in resp.json())


@pytest.mark.parametrize(
    "payload, exp",
    [
        (
            {"query": {"text": "hello there"}, "n_neighbors": 20, "service": "clinc1"},
            "hello",
        ),
        (
            {"query": {"text": "hello there"}, "n_neighbors": 10, "service": "clinc2"},
            "hello",
        ),
        (
            {"query": {"text": "thanks"}, "n_neighbors": 10, "service": "clinc1"},
            "thanks",
        ),
    ],
)
def test_basic_post_multi_server(clinc_service, payload, exp):
    """Confirm that the basic queries come back right"""
    app = create_multi_app(clinc1=clinc_service, clinc2=clinc_service)
    client = TestClient(app)
    resp = client.post("/query", json=payload)
    print(resp.text)
    print(resp.json())
    assert resp.status_code == 200
    assert len(resp.json()) == payload["n_neighbors"]
    assert any(exp in s["item"]["text"] for s in resp.json())
