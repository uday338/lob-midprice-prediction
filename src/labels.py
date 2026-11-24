"""3-class mid-price direction labels and a leakage-safe time split."""
from __future__ import annotations

import numpy as np
import pandas as pd

DOWN, FLAT, UP = 0, 1, 2
CLASS_NAMES = ["down", "flat", "up"]


def smoothed_future_return(mid: pd.Series, k: int) -> pd.Series:
    """(mean of the next k mids - current mid) / current mid.

    This is the labelling used for FI-2010 / DeepLOB (Ntakaris et al. 2018, Zhang et al. 2019).
    Averaging the future path makes labels less sensitive to a single bid-ask bounce.
    The last k rows have no complete future window and come out NaN.
    """
    fut_mean = mid[::-1].rolling(k, min_periods=k).mean()[::-1].shift(-1)
    return (fut_mean - mid) / mid


def make_labels(ret: pd.Series, alpha: float) -> pd.Series:
    lab = pd.Series(FLAT, index=ret.index, dtype="float")
    lab[ret > alpha] = UP
    lab[ret < -alpha] = DOWN
    lab[ret.isna()] = np.nan
    return lab


def time_split(n: int, k: int, frac=(0.70, 0.15, 0.15)):
    """Chronological 70/15/15 split with a purge gap of k events at each boundary.

    Labels look k events into the future, so the last k training rows would otherwise
    have labels computed from prices inside the validation period (and the same for
    validation -> test). Dropping them is the 'purging' step from Lopez de Prado (2018).
    """
    a = int(n * frac[0])
    b = int(n * (frac[0] + frac[1]))
    train = np.arange(0, a - k)
    val = np.arange(a, b - k)
    test = np.arange(b, n - k)
    return train, val, test


def choose_alpha(ret_train: pd.Series) -> float:
    """Pick the 'flat' threshold from the TRAIN set only so the three classes are about equal.

    Using quantiles of the whole series would leak test-set information into the labels.
    """
    r = ret_train.dropna().abs()
    return float(r.quantile(1 / 3))
