import pandas as pd


def fetch_clinc():
    """
    Loads the clinc conversational intents data.

    Usage:

    ```python
    from simsity.datasets import fetch_clinc

    fetch_clinc()

    #                                          text
    # 0                              find my wallet
    # 1  can you give me the gps location of harvey
    # 2    where's my buddy steve right this second
    # 3        locate jenny at her present position
    # 4          let me know where jim is right now
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

    #                      name         suburb postcode
    # 0          khimerc thomas      charlotte    2826g
    # 1       lucille richardst     kannapolis    28o81
    # 2      reb3cca bauerboand        raleigh    27615
    # 3          maleda mccloud      goldsboro    2753o
    # 4          belida stovall     morrisvill    27560
    ```
    """
    return pd.read_csv(
        "https://raw.githubusercontent.com/koaning/simsity/main/tests/data/votes.csv"
    )
