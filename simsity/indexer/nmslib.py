import nmslib
from simsity.indexer import Indexer


class NMSlibIndexer(Indexer):
    def __init__(self, n_neighbors=5, metric="euclidean", method="sw-graph", n_jobs=1):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def index(self, data):
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
        results = self.nmslib_.knnQuery(item, k=n_neighbors)
        indices, distances = zip(*results)
        return indices, distances
