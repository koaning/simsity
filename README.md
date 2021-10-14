<img src="icon.png" width=125 height=125 align="right">

# simsity

> Super Simple Similarities Services

This repository contains simple tools to help in similarity
treival scenarios. Typical usecases include early stage bulk
labelling and duplication discovery.

```python
from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer
from sklearn.feature_extraction.text import CountVectorizer


# The Indexer handles the nearest neighbor search
# The Encoder handles the encoding of the datapoints
service = Service(
    indexer=PyNNDescentIndexer(metric="euclidean"), 
    encoder=CountVectorizer()
)

# Index the datapoints
service.train_from_csv("clinc-data.csv", text_col="text")

# Query the datapoints
service.query("give me directions", n_neighbors=100)

# Save the entire system
service.save("/tmp/simple-model")

# You can also load the model now. 
Service.load("/tmp/simple-model")
```