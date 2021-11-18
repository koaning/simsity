from fastapi import FastAPI

from pydantic import BaseModel, validator
from simsity.service import Service


class Params(BaseModel):
    """Parameters for the query endpoint."""

    query: dict
    n_neighbors: int = 5

    @validator("n_neighbors")
    def n_neighbors_must_be_positive(cls, value):
        """Gotta make sure they're positive."""
        if value <= 0:
            raise ValueError(f"we expect n_neighbors >= 0, we received {value}")
        return value


def create_app(service: Service):
    """Start a small webserver with the Service."""
    app = FastAPI()

    @app.post("/query")
    def query(params: Params):
        """The main query endpoint."""
        return service.query(**params.query, n_neighbors=params.n_neighbors)

    return app


class MultiParams(BaseModel):
    """Parameters for the query endpoint."""

    query: dict
    n_neighbors: int = 5
    service: str

    @validator("n_neighbors")
    def n_neighbors_must_be_positive(cls, value):
        """Gotta make sure they're positive."""
        if value <= 0:
            raise ValueError(f"we expect n_neighbors >= 0, we received {value}")
        return value


def create_multi_app(**services):
    """
    Creates a FastAPI app from multiple services.

    Arguments:
        services: (name, service) pairs
    """
    app = FastAPI()

    @app.post("/query")
    def query(params: MultiParams):
        """Send a query to one of multiple services"""
        service = services[params.service]
        return service.query(**params.query, n_neighbors=params.n_neighbors)

    return app


def run_multi_service(host, port=8080, **services):
    """
    Runs a single endpoint that hosts multiple services.

    Once the server is started, you can `POST` the service using the following URL:

    ```
    http://<host>:<port>/query
    ```

    Arguments:
        host: Host to bind the server to.
        port: Port to bind the server to.
    """
    import uvicorn

    uvicorn.run(create_multi_app(**services), host=host, port=port)
