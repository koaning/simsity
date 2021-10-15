import pandas as pd

from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer
from sklearn.feature_extraction.text import CountVectorizer


def test_smoke(tmpdir):
    """
    Run a simple smoke test to ensure that the service is working.
    """
    service = Service(
        indexer=PyNNDescentIndexer(metric="euclidean"), encoder=CountVectorizer()
    )

    df = pd.read_csv("tests/data/clinc-data.csv")
    service.train_text_from_dataf(df, text_col="text")

    service.query(text="give me directions", n_neighbors=100)

    service.save(tmpdir)

    reloaded = Service.load(tmpdir)

    assert reloaded._trained
