from pynndescent import NNDescent


class PyNNDescentIndexer:
    """
    An indexer based on PyNNDescent.

    Arguments:
        metric: The metric to use for the index.
        n_neighbors: The number of neighbors to use for the index.
    """

    def __init__(self, metric="euclidean", n_neighbors=10) -> None:
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.model = None

    def index(self, data):
        """
        Index the given data.

        Arguments:
            data: The data to index.
        """
        self.model = NNDescent(data, metric=self.metric, n_neighbors=self.n_neighbors)
        self.model.prepare()

    def query(self, query, n_neighbors=1):
        """
        Query the index.

        Arguments:
            query: The query to query the index with.
            n_neighbors: The number of neighbors to return.
        """
        return self.model.query(query, n_neighbors)
