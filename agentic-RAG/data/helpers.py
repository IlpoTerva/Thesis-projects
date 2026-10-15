import pandas as pd
from pathlib import Path

def load_data(data_pth) -> pd.DataFrame:
    return pd.read_csv(data_pth)

def filter_by_date(df,start, end) -> pd.DataFrame:
    """Filter a DataFrame by start and end dates."""
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return df

