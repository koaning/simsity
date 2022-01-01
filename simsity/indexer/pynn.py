import json
import pathlib
from pathlib import Path
from joblib import dump, load

from pynndescent import NNDescent

from simsity.indexer.common import Indexer


class PyNNDescentIndexer(Indexer):
    """
    An indexer based on PyNNDescent.

    Arguments:
        metric: The metric to use for the index.
        n_neighbors: The number of neighbors to use for the index.
        random_state: The random state to use for the index.
        n_jobs: The number of parallel jobs to run for neighbors index construction.
    """

    def __init__(
        self, metric="euclidean", n_neighbors=10, random_state=42, n_jobs=1
    ) -> None:
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None

    def index(self, data):
        """
        Index the given data.

        Arguments:
            data: The data to index.
        """
        self.model = NNDescent(
            data,
            metric=self.metric,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.prepare()

    def query(self, query, n_neighbors=1):
        """
        Query the index.

        Arguments:
            query: The query to query the index with.
            n_neighbors: The number of neighbors to return.
        """
        if not self.model:
            raise RuntimeError("Index not yet built.")
        idx, dist = self.model.query(query, n_neighbors)
        return list(idx[0]), list(dist[0])

    def save(self, path) -> None:
        """
        Save the indexer in a path.

        Arguments:
            path: string or pathlib.Path to folder.
        """
        # Save the index
        dump(self, pathlib.Path(path) / "indexer.joblib")

        # Save the metadata so that we have the parameters on load.
        metadata_path = Path(path) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["pynn"] = dict(
            metric=self.metric,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        metadata_path.write_text(json.dumps(metadata))

    @classmethod
    def load(cls, path) -> "Indexer":
        """
        Load the indexer in a path.

        Arguments:
            path: string or pathlib.Path to folder.
        """
        return load(pathlib.Path(path) / "indexer.joblib")
