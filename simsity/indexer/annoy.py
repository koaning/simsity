import json
import pandas as pd
from pathlib import Path
from annoy import AnnoyIndex
from scipy.sparse import spmatrix

from simsity.indexer.common import Indexer


class AnnoyIndexer(Indexer):
    """
    An indexer based on Annoy.

    Note! Annoy does not support sparse data.

    Arguments:
        metric: The metric to use for the index. Can be `angular`, `euclidean`, `manning`, `manhattan` or `dot`.
        n_trees: The number of trees to build.
        n_jobs: Degree of parallism used while training.
    """

    def __init__(
        self, metric="euclidean", n_trees=10, random_state=42, n_jobs=1
    ) -> None:
        self.metric = metric
        self.random_state = (random_state,)
        self.n_trees = n_trees
        self.model = None
        self.n_jobs = n_jobs

    def index(self, data):
        """
        Index the given data.

        Arguments:
            data: The data to index.
        """
        if isinstance(data, spmatrix):
            raise ValueError("Annoy index does not support sparse matrices.")
        self.feature_size = data.shape[1]
        self.model = AnnoyIndex(self.feature_size, self.metric)
        for i in range(data.shape[0]):
            if isinstance(data, pd.DataFrame):
                self.model.add_item(i, data.iloc[i].values)
            else:
                self.model.add_item(i, data[i])
        self.model.build(self.n_trees, n_jobs=self.n_jobs)

    def query(self, query, n_neighbors=1):
        """
        Query the index.

        Arguments:
            query: The query to query the index with.
            n_neighbors: The number of neighbors to return.
        """
        if not self.model:
            raise RuntimeError("Index not yet built.")

        if isinstance(query, pd.DataFrame):
            query = query.iloc[0].values

        idx, dist = self.model.get_nns_by_vector(
            query, n=n_neighbors, include_distances=True
        )
        return idx, dist

    def save(self, path) -> None:
        """
        Save the indexer in a path.

        Arguments:
            path: string or pathlib.Path to folder.
        """
        # Save the index
        self.model.save(str(Path(path) / "index.ann"))

        # Save the metadata so that we have the parameters on load.
        metadata_path = Path(path) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["annoy"] = dict(
            metric=self.metric,
            n_trees=self.n_trees,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            feature_size=self.feature_size,
        )
        metadata_path.write_text(json.dumps(metadata))

    @classmethod
    def load(self, path) -> "Indexer":
        """
        Load the indexer in a path.

        Arguments:
            path: string or pathlib.Path to folder.
        """
        # Load metadata as well
        metadata_path = Path(path) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())["annoy"]

        # Prepare keyword arguments
        keyword_args = dict(
            metric=metadata["metric"],
            n_trees=metadata["n_trees"],
            random_state=metadata["random_state"],
            n_jobs=metadata["n_jobs"],
        )
        # Construct an empty, but configured index, before loading from disk
        index = AnnoyIndexer(**keyword_args)
        index.model = AnnoyIndex(metadata["feature_size"], keyword_args["metric"])
        index.model.load(str(Path(path) / "index.ann"))
        return index
