import time

import pandas as pd
from embetter.text import SentenceEncoder

from simsity.datasets import fetch_recipes
from simsity.service import Service
from simsity.indexer import AnnoyIndexer
from simsity.indexer.nmslib import NMSlibIndexer


# Fetch data
df_recipes = fetch_recipes()
recipes = df_recipes['text']

# Create an indexer and encoder
encoder = SentenceEncoder()
for indexer in [AnnoyIndexer(), NMSlibIndexer()]:
    print(indexer)
    # The service combines the two into a single object.
    service = Service(indexer=indexer, encoder=encoder)

    # We can now build the service using this data.
    tic = time.time()
    service.index(recipes)
    toc = time.time()
    print(toc - tic)

    # And use it
    idx, dists = service.query("vegan", n_neighbors=20)

    res = (pd.DataFrame({"recipe": recipes})
        .iloc[idx]
        .assign(dists=dists)
    )

    # Show results
    print(res)