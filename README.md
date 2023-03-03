![](docs/landing.png)

# simsity

> Simsity is a Super Simple Similarities Service[tm]. <br>
> It's all about building a neighborhood. Literally! <br>

<br>

This repository contains simple tools to help in similarity retrieval scenarios
by making a convenient wrapper around [hnswlib](https://github.com/nmslib/hnswlib/blob/master/examples/python/EXAMPLES.md).
Typical usecases include early stage bulk labelling and duplication discovery.

## Install

You can install simsity via pip.

```
python -m pip install simsity
```

```python

# Simsity provides two functions to create/load an index
from simsity import create_index, load_index
# It also has some dataset for demos 
from simsity.datasets import fetch_recipes
# Let's use embetter for embeddings 
from embetter.text import SentenceEncoder

# Here's a list of data we'll encode/index
df_recipes = fetch_recipes()
recipes = df_recipes["text"]

# Create the (scikit-learn compatible) encoder
encoder = SentenceEncoder()

# Make an index without a path
index = create_index(recipes, encoder)
texts, dists = index.query("pork")
```

You can also provide a path and then you'll be able to store/load everything.

```python
# Make an index with a path
index = create_index(recipes, encoder, path="demo")

# Load an index from a path
loader_index = load_index(path="demo", encoder=encoder)
texts, dists = index.query("pork")
```
