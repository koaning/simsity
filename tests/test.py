from embetter.text import SentenceEncoder

from simsity import create_index, load_index
from simsity.datasets import fetch_recipes

# Fetch data
df_recipes = fetch_recipes()
recipes = df_recipes["text"][:1000]

# Create an encoder
encoder = SentenceEncoder()

# Make an index with a path
index = create_index(recipes, encoder, path="demo")
texts, dists = index.query("pork")
for text in texts:
    assert "pork" in text

# Load an index from a path
loader_index = load_index(path="demo", encoder=encoder)
texts, dists = index.query("pork")
for text in texts:
    assert "pork" in text

# You can also pass a callable as an encoder
index = create_index(recipes, lambda d: encoder.transform(d), path="demo")
texts, dists = index.query("pork")
for text in texts:
    assert "pork" in text

# This as well
loader_index = load_index(path="demo", encoder=lambda d: encoder.transform(d))
texts, dists = index.query("pork")
for text in texts:
    assert "pork" in text
