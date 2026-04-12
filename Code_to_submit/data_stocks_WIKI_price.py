import numpy as np
import pandas as pd


def get_cleaned_data(file_path: str,
                     train_ratio: float = 0.70,
                     min_price: float = 3.0,
                     max_consec_zeros: int = 20) -> pd.DataFrame:
    """Load and clean WIKI prices data, returning complete return series only.

    Filters applied (no look-ahead bias):
      1. min_price      : exclude stocks with adj_close < min_price on first test day
      2. max_consec_zeros: exclude stocks with more than max_consec_zeros consecutive
                           zero returns (proxy for zero-volume / delisted periods)
    """
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

    # ── Filter 1: price > min_price on first day of test set ─────────────────
    split_idx       = int(train_ratio * len(real_time_series))
    first_test_date = real_time_series.index[split_idx]
    price_on_day    = adj_close.loc[first_test_date].reindex(real_time_series.columns)
    valid_price     = price_on_day[price_on_day >= min_price].index
    real_time_series = real_time_series[valid_price]

    # ── Filter 2: no more than max_consec_zeros consecutive zero returns ──────
    def _max_consec_zeros(s):
        is_zero = (s == 0)
        groups  = (is_zero != is_zero.shift()).cumsum()
        return (is_zero * (is_zero.groupby(groups).cumcount() + 1)).max()

    max_consec   = real_time_series.apply(_max_consec_zeros)
    valid_consec = max_consec[max_consec <= max_consec_zeros].index
    real_time_series = real_time_series[valid_consec]

    print(f"[get_cleaned_data] Universe after filters : {real_time_series.shape[1]} stocks "
          f"(price≥${min_price} on {first_test_date.date()}, "
          f"max consec zeros ≤ {max_consec_zeros})")

    return real_time_series
