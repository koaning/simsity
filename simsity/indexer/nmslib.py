import json
from pathlib import Path 
import nmslib
from simsity.indexer import Indexer


class NMSlibIndexer(Indexer):
    """
    An indexer based on NMSLib.
    """

    def __init__(self, metric="cosine", method="hnsw"):
        self.method = method
        self.metric = metric
        self.lookup = {
            "euclidean": "l2",
            "cosine": "cosinesimil",
            "l1": "l1",
            "l2": "l2",
        }
        index.model_ = None

    def index(self, data):
        """
        Index the given data.

        Arguments:
            data: The data to index.
        """
        # see more metric in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual
        space = self.lookup[self.metric]

        self.model_ = nmslib.init(method=self.method, space=space)
        self.model_.addDataPointBatch(data)
        self.model_.createIndex()
        return self

    def query(self, item, n_neighbors=1):
        """
        Query the index.

        Arguments:
            query: The query to query the index with.
            n_neighbors: The number of neighbors to return.
        """
        if not self.model_:
            raise RuntimeError("Index not yet built.")
        indices, distances = self.model_.knnQuery(item, k=n_neighbors)
        return indices, distances

    def save(self, path):
        # Save the index
        self.model_.saveIndex(str(Path(path) / "index.nmslib"), save_data=True)

        # Save the metadata so that we have the parameters on load.
        metadata_path = Path(path) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["mnslib"] = dict(
            metric=self.metric,
            method=self.method,
        )
        metadata_path.write_text(json.dumps(metadata))

    def load(self, path):
        metadata_path = Path(path) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())["mnslib"]

        # Prepare keyword arguments
        keyword_args = dict(
            metric=metadata["metric"],
            method=metadata["method"],
        )
        # Construct an empty, but configured index, before loading from disk
        index = NMSlibIndexer(**keyword_args)
        index.model_ = nmslib.init(method=index.method, space=index.lookup[index.method])
        index.model_.loadIndex(path, load_data=True)
