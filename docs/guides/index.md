The goal of this tool is to offer you a simple similarity service.

So let's build an example! We'll build a similarity searching tool for a text dataset.

## Install

In this tutorial we will leverage pretrained language models that are provided
via embetter. We will install that, together with simsity.

```
python -m pip install simsity "embetter[sentence-tfm]"
```

The install can take a while, because we are downloading PyTorch.

## Example Data

For this tutorial we will explore a set of recipe names.

```python
from simsity.datasets import fetch_recipes

df_recipes = fetch_recipes()
```

Here's what the top 5 rows look like.

| text                                        |
|:--------------------------------------------|
| pork chop noodle soup                       |
| 5 ingredient almond cake with fresh berries |
| shrimp cakes                                |
| chili roasted okra                          |
| slow cooker chicken chili                   |

The goal is to be able to look for recipes without resorting to
strict token equality. If we look for "meat" we'd like to see recipes
that have meat in it, even if the word "meat" does not appear in the
recipe title.

We will work with a list of texts in this tutorial.

```python
recipes = list(df_recipes['text'])
```

## The Tactic

The idea is that we're going to split the problem of similarity search into
two subproblems.

1. The first problem is **encoding**. If we're going to use similarities,
we'll need some way to turn our data into a numeric representation. Without
a numeric representation it'll be quite hard to compare items numerically.
2. The second problem is **indexing**. Even when we have numeric representations
to compare against, we don't want to compare *all* the possible solutions
out there. Instead we'd prefer to index out data such that it's fast to retreive.

To solve the first problem, simsity likes to re-use tools from the scikit-learn
ecosystem. An encoder in simsity is simply a scikit-learn component/pipeline that transforms
data. To solve the second problem, simsity wraps around existing tools for approximate
nearest-neighbor lookup. The goal of simsity is to combine an encoder and an indexer
into a service that's convenient for interaction.

### Example Encoder

We will now define a scikit-learn compatible encoder, which is provided
by embetter in this case.

```python
from embetter.text import SentenceEncoder

# The encoder defines how we encode the data going in.
encoder = SentenceEncoder()
```

### Example Indexer

Simsity provides indexers by wrapping around existing solutions. In particular
it supports [annoy](https://github.com/spotify/annoy) out of the box.  If you're
curious to learn how it works, you may appreciate this [segment on calmcode](https://calmcode.io/annoy/intro.html).

```python
from simsity.indexer import AnnoyIndexer

# Create an indexer
indexer = AnnoyIndexer()
```

There are many [distance metrics](https://github.com/spotify/annoy#full-python-api)
that annoy supports; `angular`, `euclidean`, `manhattan`, `hamming` and `dot`.

!!! note

    There are differences between indexers. Annoy is flexible, but you may need to add many
    trees for it to become accurate. It also only supports sparse arrays. The `PyNNDescentIndexer` indexer
    supports dense arrays as well as sparse ones! The only downside is that it does take
    a while to index all the data.


## Building a Service

Once you have an encoder and an indexer, you can construct a service.

```python
from simsity.service import Service

# The service combines the two into a single object.
service = Service(indexer=indexer, encoder=encoder)
```

This service can now index on your dataset. It will start by first training
the encoder pipeline. After that the data will be transformed and indexed
by the indexer. All of this will be handled by the following call:

```python
service.index(recipes)
```

It's good to notice that we're being explicit here about which features
are being used. We're telling our service to only consider the `"text"` column!
This is important when you want to query your data.

### Query the Data

You can now try to look for similar items by sending a query to the service.
Note that the keyword argument `text=` corresponds with the features that
we chose to index earlier.

```python
# Get indices and distances
idx, dists = service.query(text="meat", n_neighbors=10, out="dataframe")

# Show as pandas dataframe
import pandas as pd
pd.DataFrame({"recipe": recipes}).iloc[idx].assign(dists=dists)
```

This is the table that you'll get back.

| recipe                    |    dists |
|:--------------------------|---------:|
| roast beef                | 0.840229 |
| beef stew                 | 0.851083 |
| buffalo chicken meatballs | 0.853497 |
| meat feast pizza          | 0.873777 |
| chicken marsala meatballs | 0.874075 |
| meat dim sum              | 0.886897 |
| moroccan meatballs        | 0.892015 |
| meatball sub sandwich     | 0.899369 |
| italian meatballs         | 0.899428 |
| juicy italian meatballs   | 0.906681 |

The quality of what you get back depends on the data that you give the system,
the encoding and the indexer that you pick. You can probably improve the results
by picking an angular distance measure in annoy, but you could also try out other
embedding models as well.

The goal of this package is to make it easy to interact and experiment
with the idea of "building neigborhoods of similar items". Hence the name: simsity.

## All Code

Here's the full code block that we've used in this section.

```python
import pandas as pd
from embetter.text import SentenceEncoder

from simsity.datasets import fetch_recipes
from simsity.service import Service
from simsity.indexer import AnnoyIndexer


# Fetch data
df_recipes = fetch_recipes()
recipes = df_recipes['text']

# Create an indexer
indexer = AnnoyIndexer()

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
