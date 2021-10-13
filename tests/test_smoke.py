import pandas as pd

from simwity.service import Service
from simwity.indexer import PyNNDescentIndexer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from whatlies.language import BytePairLanguage

def test_smoke():
    service = Service(
        indexer=PyNNDescentIndexer(metric="euclidean"), 
        encoder=BytePairLanguage()
    )

    service.train_from_csv("clinc-data.csv", text_col="text")

    service.query("give me directions", n_neighbors=100)

    service.save("/tmp/simple-model")
