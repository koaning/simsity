import json
from pathlib import Path
import nmslib
from simsity.indexer import Indexer


class NMSlibIndexer(Indexer):
    """
    An indexer based on NMSLib.
    """

    def __init__(self, metric="cosine", method="hnsw"):
        self.metric = metric
        self.method = method
        self.lookup = {
            "euclidean": "l2",
            "cosine": "cosinesimil",
            "l1": "l1",
            "l2": "l2",
        }

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
        if not getattr(self, "model_", None):
            raise RuntimeError("Must index before sending query.")
        indices, distances = self.model_.knnQuery(item, k=n_neighbors)
        return indices, distances

    def save(self, path):
        """Save the model state to disk"""
        # Save the index
        self.model_.saveIndex(str(Path(path) / "index.nmslib"), save_data=True)

        # Save the metadata so that we have the parameters on load.
        metadata_path = Path(path) / "metadata.json"
        metadata = {}
        metadata["mnslib"] = dict(
            metric=self.metric,
            method=self.method,
        )
        metadata_path.write_text(json.dumps(metadata))

    @classmethod
    def load(cls, path):
        """Load the model state from disk"""
        metadata_path = Path(path) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())["mnslib"]

        # Prepare keyword arguments
        keyword_args = dict(
            metric=metadata["metric"],
            method=metadata["method"],
        )
        # Construct an empty, but configured index, before loading from disk
        index = NMSlibIndexer(**keyword_args)
        space = index.lookup[index.metric]
        index.model_ = nmslib.init(method=index.method, space=space)
        index.model_.loadIndex(str(path / "index.nmslib"), load_data=True)
        index.model_.createIndex()
        return index
