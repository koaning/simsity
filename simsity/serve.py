from fastapi import FastAPI
from simsity.service import Service


def create_app(service: Service):
    """Start a small webserver with the Service."""
    app = FastAPI()

    @app.post("/query")
    def query(params: dict, n_neighbors: int = 5):
        """The main query endpoint."""
        params = {**params, "n_neighbors": n_neighbors}
        return service.query(**params)

    return app
