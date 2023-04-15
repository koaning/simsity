from embetter.text import SentenceEncoder

from simsity import create_index, load_index
from simsity.datasets import fetch_recipes

# Fetch data
df_recipes = fetch_recipes()
recipes = df_recipes["text"][:1000]

# Create an encoder
encoder = SentenceEncoder()


def check_output(texts):
    for text in texts:
        assert "pork" in text


def test_base_usage(tmpdir):
    # Make an index with a path
    index = create_index(recipes, encoder, path=tmpdir)
    out1, _ = index.query("pork")
    check_output(out1)

    # Load an index from a path
    loader_index = load_index(path=tmpdir, encoder=encoder)
    out2, _ = loader_index.query("pork")
    check_output(out2)
    assert out1 == out2


def test_callable_usage(tmpdir):
    # You can also pass a callable as an encoder
    index = create_index(recipes, lambda d: encoder.transform(d), path=tmpdir)
    out1, _ = index.query("pork")
    check_output(out1)

    # This as well
    loader_index = load_index(path=tmpdir, encoder=encoder)
    out2, _ = loader_index.query("pork")
    check_output(out2)
    assert out1 == out2
