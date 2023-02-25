import srsly
from typing import List
from pathlib import Path
import warnings
import orjson
from simsity.indexer import Indexer
from sklearn.base import TransformerMixin
from skops.io import dump, load

from simsity.indexer import AnnoyIndexer, PyNNDescentIndexer, NMSlibIndexer


class Service:
    """
    This object represents a nearest neighbor lookup service. You can
    pass it an encoder and a method to index the data.
    """

    def __init__(self, encoder, indexer, data):
        self.encoder = encoder
        self.indexer = indexer
        self.data = dict(enumerate(data))

    def index(self):
        """
        Indexes the service from a dataframe.

        Arguments:
            df: Pandas DataFrame that contains text to train the service with.
            features: Names of the features to encode.
        """
        try:
            data = self.encoder.transform(list(self.data.values()))
        except Exception as ex:
            warnings.warn(
                "Encountered error using pretrained encoder. Are you sure it is trained?"
            )
            raise ex

        self.indexer.index(data)
        return self

    def query(self, item, n_neighbors=10):
        """
        Query the service.

        Arguments:
            n_neighbors: Number of neighbors to return.
            out: Output format. Can be either "list" or "dataframe".
            kwargs: Arguments to pass as the query.
        """
        data = self.encoder.transform([item])
        idx, dist = self.indexer.query(data[0], n_neighbors=n_neighbors)
        return [self.data[i] for i in idx], dist

    def to_disk(self, path):
        """
        Writes the service to disk.

        Arguments:
            path: folder to write service state into
        """
        path = Path(path)
        metadata = {"indexer": self.indexer.__class__.__name__}
        srsly.write_json(path / "metadata.json", metadata)
        self.indexer.to_disk(path)
        dump(self.encoder, path / "encoder.skops")
        print(self.data)
        srsly.write_json(path / "data.json", self.data)

    @staticmethod
    def _load_indexer(metadata) -> Indexer:
        indexer_str = metadata["indexer"].lower()
        if "annoy" in indexer_str:
            return AnnoyIndexer
        if "pynn" in indexer_str:
            return PyNNDescentIndexer
        if "nms" in indexer_str:
            return NMSlibIndexer
        raise RuntimeError("Did not recognize indexer from {indexer_str}.")
    
    @classmethod
    def from_disk(cls, path):
        """
        Loads the service from disk.

        Arguments:
            path: folder to write service from
        """
        path = Path(path)
        encoder = load(path / "encoder.skops")
        metadata = srsly.read_json(path / "metadata.json")
        indexer = cls._load_indexer(metadata=metadata)
        indexer.from_disk(path)
        data = srsly.read_json(path / "data.json")
        return Service(encoder, indexer, data)
