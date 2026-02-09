import numpy as np
import pandas as pd


def get_cleaned_data(file_path: str) -> pd.DataFrame:
    """Load and clean WIKI prices data, returning complete return series only."""
    data = pd.read_csv(file_path, parse_dates=["date"])

    data = data[data["volume"] > 0]

    adj_close = data.pivot(index="date", columns="ticker", values="adj_close")
    adj_close = adj_close.sort_index()

    adj_close = adj_close.loc["2000-01-01":]

    adj_close = adj_close[adj_close.isna().sum(axis=1) < 2000]

    adj_close = adj_close.loc[:, ~adj_close.isna().all(axis=0)]

    adj_ret = adj_close.ffill().pct_change()

    mask = (~adj_close.ffill().isna()) & (~adj_close.bfill().isna())
    adj_ret = adj_ret.where(mask, np.nan).iloc[1:]

    real_time_series = adj_ret.dropna(axis=1)

    return real_time_series
