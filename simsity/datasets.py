import pandas as pd


def fetch_clinc():
    """
    Loads the clinc conversational intents data.

    Usage:

    ```python
    from simsity.datasets import fetch_clinc

    fetch_clinc()
    ```
    """
    return pd.read_csv(
        "https://raw.githubusercontent.com/koaning/simsity/main/tests/data/clinc-data.csv"
    )


def fetch_voters():
    """
    Loads the voters dataset.

    Usage:

    ```python
    from simsity.datasets import fetch_voters

    fetch_voters()
    ```
    """
    return pd.read_csv(
        "https://raw.githubusercontent.com/koaning/simsity/main/tests/data/votes.csv"
    )
