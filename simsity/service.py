import warnings

from simsity.preprocessing import Identity
from simsity.indexer import Indexer


class Service:
    """
    This object represents a nearest neighbor lookup service. You can
    pass it an encoder and a method to index the data.

    Arguments:
        encoder: A scikit-learn compatible encoder for the input.
        indexer: A compatible indexer for the nearest neighbor search.
    """

    def __init__(self, encoder=Identity(), indexer: Indexer = None) -> None:
        self.encoder = encoder
        self.indexer = indexer

    def index(self, X):
        """
        Indexes the service from a dataframe.

        Arguments:
            df: Pandas DataFrame that contains text to train the service with.
            features: Names of the features to encode.
        """
        try:
            data = self.encoder.transform(X)
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
        return idx, dist
