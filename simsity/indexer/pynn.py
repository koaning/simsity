from pynndescent import NNDescent

from simsity.indexer.common import Indexer


class PyNNDescentIndexer(Indexer):
    """
    An indexer based on PyNNDescent.

    Arguments:
        metric: The metric to use for the index.
        n_neighbors: The number of neighbors to use for the index.
        random_state: The random state to use for the index.
        n_jobs: The number of parallel jobs to run for neighbors index construction. `None` means 1 while `-1` means all processors.
    """

    def __init__(
        self, metric="euclidean", n_neighbors=10, random_state=42, n_jobs=None
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
