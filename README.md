<img src="icon.png" width=125 height=125 align="right">

# simsity

> Simsity is a Super Simple Similarities Service[tm]. <br>
> It's all about building a neighborhood. Literally!

This repository contains
simple tools to help in similarity retreival scenarios by making a convient
wrapper around encoding strategies as well as nearest neighbor approaches. 
Typical usecases include early stage bulk labelling and duplication discovery.

## Warning

Alpha software. Expect things to break. Do not use in production.

## Example

This is the basic setup for this package.

```python
import pandas as pd
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
df = pd.read_csv("tests/data/clinc-data.csv")
service.train_text_from_dataf(df, text_col='text')

# Query the datapoints
service.query_text("bank credit", n_neighbors=20)

# Save the entire system
service.save("/tmp/simple-model")

# You can also load the model now.
Service.load("/tmp/simple-model")
```
