"""Uruchomienie z katalogu głównego projektu:
    uv run python -m exercises.linear_regression.data"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"

def load_frame() -> pd.DataFrame:
    """ Cały zbiór California Housing jako jeden DataFrame ."""
    #fetch_california_housing(as_frame=True) zwraca kolumnę celu MedHouseVal.
    housing = fetch_california_housing(as_frame=True)
    return housing.frame

def split_data(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Zwraca (train, val, test) w proporcjach 60/20/20. """
    rest, test = train_test_split(df, test_size=0.2, random_state=seed)
    train, val = train_test_split(rest, test_size=0.25, random_state=seed)
    return train, val, test

if __name__ == "__main__":
    train, val, test = split_data(load_frame())
    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
    #Dla kontroli: 12 384 wierszy treningowych, 4 128 walidacyjnych i 4 128 testowych.