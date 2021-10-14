from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer
from sklearn.feature_extraction.text import CountVectorizer


def test_smoke():
    service = Service(
        indexer=PyNNDescentIndexer(metric="euclidean"), 
        encoder=CountVectorizer()
    )

    service.train_from_csv("clinc-data.csv", text_col="text")

    service.query("give me directions", n_neighbors=100)

    service.save("/tmp/simple-model")
