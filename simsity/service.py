import json
import pathlib

import pandas as pd
from joblib import dump, load
from simsity import __version__


class Service:
    """
    This object represents a nearest neighbor lookup service. You can
    pass it an encoder and a method to index the data.

    Arguments:
        encoder: A scikit-learn compatible encoder for the input.
        indexer: A compatible indexer for the nearest neighbor search.
        storage: A dictionary containing the data to be retreived with index. Meant to be ignored by humans.

    Usage:

    ```python
    from simsity.service import Service
    from simsity.indexer import PyNNDescentIndexer
    from sklearn.feature_extraction.text import CountVectorizer


    service = Service(
        encoder=CountVectorizer(),
        indexer=PyNNDescentIndexer(metric="euclidean")
    )
    ```
    """

    def __init__(self, encoder, indexer, storage=None) -> None:
        self.encoder = encoder
        self.indexer = indexer
        self.storage = storage if storage else {}
        self._trained = False

    def train_text_from_dataf(self, df, text_col="text"):
        """
        Trains the service from a dataframe assuming text as input.

        Arguments:
            df: Pandas DataFrame that contains text to train the service with.
            text_col: Name of the column containing text.

        Usage:

        ```python
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer

        from simsity.service import Service
        from simsity.indexer import PyNNDescentIndexer


        service = Service(
            encoder=CountVectorizer(),
            indexer=PyNNDescentIndexer(metric="euclidean")
        )

        df = pd.read_csv("tests/data/clinc-data.csv").head(100)
        service.train_text_from_dataf(df, text_col="text")
        ```
        """
        texts = list(df[text_col])
        self.storage = {i: {"text": t} for i, t in enumerate(texts)}
        data = self.encoder.fit_transform(texts)
        self.indexer.index(data)
        self._trained = True
        return self

    def train_from_dataf(self, df, features=None):
        """
        Trains the service from a dataframe.

        Arguments:
            df: Pandas DataFrame that contains text to train the service with.
            features: Names of the features to encode.

        Usage:

        ```python
        import pandas as pd

        from simsity.service import Service
        from simsity.indexer import PyNNDescentIndexer
        from dirty_cat import GapEncoder

        df = pd.read_csv("tests/data/votes.csv")

        service = Service(
            indexer=PyNNDescentIndexer(metric="euclidean"),
            encoder=GapEncoder()
        )

        service.train_from_dataf(df)
        res = service.query(name="khimerc thmas", suburb="chariotte", postcode="28273", n_neighbors=3)
        pd.DataFrame([{**r['item'], 'dist': r['dist']} for r in res])
        ```
        """
        subset = df
        if features:
            subset = df[features]
        self.storage = {i: r for i, r in enumerate(subset.to_dict(orient="records"))}
        data = self.encoder.fit_transform(subset)
        self.indexer.index(data)
        self._trained = True
        return self

    def query_text(self, text, n_neighbors=10):
        """
        Query the service

        ```python
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer

        from simsity.service import Service
        from simsity.indexer import PyNNDescentIndexer


        service = Service(
            encoder=CountVectorizer(),
            indexer=PyNNDescentIndexer(metric="euclidean")
        )

        df = pd.read_csv("tests/data/clinc-data.csv").head(100)
        service.train_text_from_dataf(df, text_col="text")
        service.query_text("Hello there", n_neighbors=10)
        ```
        """
        data = self.encoder.transform([text])
        idx, dist = self.indexer.query(data, n_neighbors=n_neighbors)
        return [
            {"item": self.storage[idx[0][i]], "dist": dist[0][i]} for i in range(idx)
        ]

    def query(self, n_neighbors=10, out="list", **kwargs):
        """
        Query the service
        """
        if not self._trained:
            raise RuntimeError("Cannot save, Service is not trained.")
        data = self.encoder.transform(pd.DataFrame([{**kwargs}]))
        idx, dist = self.indexer.query(data[0], n_neighbors=n_neighbors)
        res = [{"item": self.storage[idx[i]], "dist": dist[i]} for i in range(len(idx))]
        if out == "list":
            return res
        if out == "dataframe":
            return pd.DataFrame([{**r["item"], "dist": r["dist"]} for r in res])

    def save(self, path):
        """
        Save the service

        Arguments:
            path: Path to the folder to save the service to.
        """
        if not self._trained:
            raise RuntimeError("Cannot save, Service is not trained.")
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        storage_path = pathlib.Path(path) / "storage.json"
        storage_path.write_text(json.dumps(self.storage))
        metadata_path = pathlib.Path(path) / "metadata.json"
        metadata_path.write_text(json.dumps({"version": __version__}))
        dump(self.encoder, pathlib.Path(path) / "encoder.joblib")
        dump(self.indexer, pathlib.Path(path) / "indexer.joblib")

    @classmethod
    def load(cls, path):
        """
        Loads a service

        Arguments:
            path: Path to the folder to load the service from.
        """
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"{path} does not exist")
        metadata_path = pathlib.Path(path) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata["version"] != __version__:
            raise RuntimeError(
                f"Version mismatch. Expected {__version__}, got {metadata['version']}"
            )
        storage_path = pathlib.Path(path) / "storage.json"
        storage = {int(k): v for k, v in json.loads(storage_path.read_text()).items()}
        encoder = load(pathlib.Path(path) / "encoder.joblib")
        decoder = load(pathlib.Path(path) / "indexer.joblib")
        service = cls(encoder, decoder, storage)
        service._trained = True
        return service
