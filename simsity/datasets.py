import pandas as pd


def load_clinc():
    """
    Loads the clinc conversational intents data.
    """
    return pd.read_csv(
        "https://raw.githubusercontent.com/koaning/simsity/main/tests/data/clinc-data.csv"
    )
