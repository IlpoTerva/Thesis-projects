import pandas as pd
from enum import Enum

class Datatype(str, Enum):
    faults= "faults"
    operations = "operations"

def load_data(data: Datatype) -> pd.DataFrame:
    df = pd.read_csv(f"mes/mes_{data.value}.csv", parse_dates=["date"])
    return df

def filter_by_date(df, mask, start, end) -> pd.DataFrame:
    """Filter a DataFrame by start and end dates."""
    if start:
        mask &= df["date"] >= pd.Timestamp(start)
    if end:
        mask &= df["date"] <= pd.Timestamp(end) + pd.Timedelta(days=1)
    return mask, df

