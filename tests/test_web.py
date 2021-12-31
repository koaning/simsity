import pytest

from simsity.serve import create_app
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

    assert resp.status_code == 200
    assert len(resp.json()) == payload["n_neighbors"]
    assert any(exp in s["item"]["text"] for s in resp.json())
