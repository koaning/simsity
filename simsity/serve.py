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
