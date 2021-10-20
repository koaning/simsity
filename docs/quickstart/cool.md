<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<script src="https://cdn.jsdelivr.net/gh/koaning/justcharts/justcharts.js"></script>

This document explores some cool/useful tricks you can pull off with this library.

## Benchmarking

Before diving into benchmarking, we should be acknowledge that coming
up with meaningful benchmarks is hard. The goal of this document is to
inspire folks to think about benchmarking, not to suggest that this page
highlights a state of the art result.

Having said that ... let's say that you're interested in building a retreival service and
you happen to have a dataset that's labelled. In that case you may
be able to calculate precision-at-k and recall-at-k ([wiki](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_k)). The [examples/benchmark.ipynb]() file on the
GitHub repository shows a full example using the clinc-dataset.

The benchmark compares two encoders. One is fairly basic and only
tracks word-tokens while the other includes subword [embeddings](https://rasahq.github.io/whatlies/api/language/bpemb_lang/) and
countvectors.

```python
# Original Encoder Pipeline
encoder = make_pipeline(
    ColumnLister('text'),
    CountVectorizer()
)

# New Encoder Pipeline
encoder = make_pipeline(
    ColumnLister('text'),
    make_union(
        CountVectorizer(),
        CountVectorizer(analyzer="char", ngram_range=(2, 3)),
        BytePairLanguage("en", vs=1_000),
        BytePairLanguage("en", vs=100_000),
    )
)
```

The results from the comparison are summarised in the chart below.
Feel free to click/drag/hover/zoom. You can double-click to reset the
view.

<vegachart schema-url="../pretty-chart.json"></vegachart>

Again, we don't want to suggest that the encoders that we used are
state of the art, but we do hope the notebook offers a convenient
starting point for folks to start benchmarking experiments.

## Interactive Widgets

For extra interactivity you may be interested in using simsity with
interactive jupyter widgets. To use these, you'll want to double
check that you're using a modern jupyterlab installation and that
the `ipywidgets` library is installed.

```
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

You can now re-use a service as an interactive widget.

```python
import ipywidgets as widgets

def reduce(q):
    subset = service.query(text=q, n_neighbors=15, out="dataframe")
    display(subset)

q = widgets.Text()

out = widgets.interactive_output(reduce, {'q': q})

widgets.VBox([q, out])
```

If you're unfamiliar with the widgets and appreciate a course we recommend
checking [this calmcode.io course](https://calmcode.io/ipywidgets/introduction.html).
