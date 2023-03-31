
![landing](https://user-images.githubusercontent.com/1019791/222645884-fd88cd66-3dd0-4b6e-98f4-65586040e538.png)

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

The goal of simsity is to be minimal, to make rapid prototyping very easy and to be "just enough" for medium sized datasets. You will mainly interact with these two functions. 

```python
from simsity import create_index, load_index
```

As their names imply, you can use these to create an index or to load one from disk. 

## Quickstart

```python
from simsity import create_index, load_index

# Let's fetch some demo data
from simsity.datasets import fetch_recipes
df_recipes = fetch_recipes()
recipes = df_recipes["text"]

# Let's use embetter for embeddings 
from embetter.text import SentenceEncoder
encoder = SentenceEncoder()

# Populate the ANN vector index and use it. 
index = create_index(recipes, encoder)
texts, dists = index.query("pork")

# You can also query using vectors
v_pork = encoder.transform(["pork"])[0]
texts, dists = index.query_vector(v_pork)
```

You can also provide a path and then you'll be able to store/load everything.

```python
# Make an index with a path
index = create_index(recipes, encoder, path="demo")

# Load an index from a path
reloaded_index = load_index(path="demo", encoder=encoder)
texts, dists = reloaded_index.query("pork")
```

That's it! Happy hacking!
