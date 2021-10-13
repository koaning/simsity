import pandas as pd

from simwity.service import Service
from simwity.indexer import PyNNDescentIndexer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from whatlies.language import UniversalSentenceLanguage

service = Service(
    indexer=PyNNDescentIndexer(metric="euclidean"), 
    encoder=UniversalSentenceLanguage()
)

service.train_from_csv("clinc-data.csv", text_col="text")

service.query("give me directions", n_neighbors=100)

service.save("/tmp/simple-model")
