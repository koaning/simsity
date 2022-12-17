import nmslib
from simsity.indexer import Indexer


class NMSlibIndexer(Indexer):
    """
    An indexer based on NMSLib.
    """

    def __init__(self, n_neighbors=5, metric="euclidean", method="sw-graph", n_jobs=1):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def index(self, data):
        """
        Index the given data.

        Arguments:
            data: The data to index.
        """
        # see more metric in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual
        space = {
            "euclidean": "l2",
            "cosine": "cosinesimil",
            "l1": "l1",
            "l2": "l2",
        }[self.metric]

        self.nmslib_ = nmslib.init(method=self.method, space=space)
        self.nmslib_.addDataPointBatch(data)
        self.nmslib_.createIndex()
        return self

    def query(self, item, n_neighbors=1):
        """
        Query the index.

        Arguments:
            query: The query to query the index with.
            n_neighbors: The number of neighbors to return.
        """
        results = self.nmslib_.knnQuery(item, k=n_neighbors)
        indices, distances = zip(*results)
        return indices, distances
