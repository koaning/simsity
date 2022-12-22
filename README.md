![](docs/landing.png)

# simsity

> Simsity is a Super Simple Similarities Service[tm]. <br>
> It's all about building a neighborhood. Literally! <br>

<br>

This repository contains simple tools to help in similarity retrieval scenarios
by making a convenient wrapper around encoding strategies as well as nearest neighbor
approaches. Typical usecases include early stage bulk labelling and duplication discovery.

## Install

You can install simsity via pip.

```
python -m pip install simsity
```

It's usually recommended that you also install [embetter](https://github.com/koaning/embetter).
## Quickstart

This is the basic setup for this package.

```python
import pandas as pd
from embetter.text import SentenceEncoder

from simsity.datasets import fetch_recipes
from simsity.service import Service
from simsity.indexer import AnnoyIndexer


# Fetch data
df_recipes = fetch_recipes()
recipes = df_recipes['text']

# Create an indexer and encoder
indexer = AnnoyIndexer()
encoder = SentenceEncoder()

# The service combines the two into a single object.
service = Service(indexer=indexer, encoder=encoder)

# We can now build the service using this data.
service.index(recipes)

# And use it
idx, dists = service.query("meat", n_neighbors=10)

res = (pd.DataFrame({"recipe": recipes})
    .iloc[idx]
    .assign(dists=dists)
    .to_markdown(index=False)
)

# Show results
print(res)
```
