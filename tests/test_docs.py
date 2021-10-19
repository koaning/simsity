import pytest
import pathlib
from mktestdocs import check_docstring, get_codeblock_members, grab_code_blocks

from simsity.datasets import fetch_clinc, fetch_voters
from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer


@pytest.mark.parametrize(
    "obj", get_codeblock_members(Service), ids=lambda d: d.__qualname__
)
def test_service_members(obj):
    """Test methods of the `Service` class."""
    check_docstring(obj)


@pytest.mark.parametrize(
    "obj", get_codeblock_members(PyNNDescentIndexer), ids=lambda d: d.__qualname__
)
def test_indexer_members(obj):
    """Test methods of the `Indexer` class."""
    check_docstring(obj)


@pytest.mark.parametrize("func", [fetch_clinc, fetch_voters], ids=lambda d: d.__name__)
def test_function_docstrings(func):
    """Test the docstring code of some functions."""
    check_docstring(obj=func)


def test_readme_file():
    """Test the README file."""
    grab_code_blocks(pathlib.Path("README.md").read_text())
