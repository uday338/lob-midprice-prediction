"""
36 microstructure features computed from a LOBSTER DataFrame (see lob_io.load_lobster).

Every feature at row t uses only information available right after message t
(the book state after t and messages <= t). Rolling windows are in *events*, not seconds.

Feature groups (count):
  A. absolute price level ............ 5   <- the leaky group, dropped in the final model
  B. spread and book shape ........... 8
  C. depth and imbalance ............. 7
  D. order flow imbalance (OFI) ...... 5
  E. message flow .................... 7
  F. returns and volatility .......... 4
                                      ---
                                       36
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ABSOLUTE_PRICE_FEATURES = ["mid_price", "best_bid", "best_ask", "microprice", "book_vwap_5"]


def _roll_sum(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=1).sum()


def _ofi_level(df: pd.DataFrame, i: int) -> pd.Series:
    """Order flow imbalance at level i (Cont, Kukanov & Stoikov 2014), one value per event.

    e_n = q^b_n 1{P^b_n >= P^b_{n-1}} - q^b_{n-1} 1{P^b_n <= P^b_{n-1}}
        - q^a_n 1{P^a_n <= P^a_{n-1}} + q^a_{n-1} 1{P^a_n >= P^a_{n-1}}
    Positive = net buying pressure (bids growing / asks shrinking).
    """
    pb, qb = df[f"bid_p{i}"], df[f"bid_s{i}"]
    pa, qa = df[f"ask_p{i}"], df[f"ask_s{i}"]
    pb0, qb0, pa0, qa0 = pb.shift(), qb.shift(), pa.shift(), qa.shift()
    e = (
        qb * (pb >= pb0) - qb0 * (pb <= pb0)
        - qa * (pa <= pa0) + qa0 * (pa >= pa0)
    )
    return e.fillna(0.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    bid1, ask1 = df["bid_p1"], df["ask_p1"]
    bq1, aq1 = df["bid_s1"].astype(float), df["ask_s1"].astype(float)
    mid = (bid1 + ask1) / 2
    spread = ask1 - bid1
    micro = (ask1 * bq1 + bid1 * aq1) / (bq1 + aq1)

    # ---------- A. absolute price level (leaky across a time split)
    f["mid_price"] = mid
    f["best_bid"] = bid1
    f["best_ask"] = ask1
    f["microprice"] = micro
    pv = sum(df[f"bid_p{i}"].fillna(0) * df[f"bid_s{i}"] + df[f"ask_p{i}"].fillna(0) * df[f"ask_s{i}"] for i in range(1, 6))
    vv = sum(df[f"bid_s{i}"] + df[f"ask_s{i}"] for i in range(1, 6))
    f["book_vwap_5"] = pv / vv

    # ---------- B. spread and book shape (all relative to the mid, so price-level free)
    f["spread"] = spread
    f["spread_bps"] = spread / mid * 1e4
    f["microprice_dev"] = (micro - mid) / spread  # in [-0.5, 0.5]
    f["wmid5_dev"] = (f["book_vwap_5"] - mid) / spread
    f["ask_gap_12"] = (df["ask_p2"] - ask1).fillna(0)
    f["bid_gap_12"] = (bid1 - df["bid_p2"]).fillna(0)
    ask_depth5 = sum(df[f"ask_s{i}"] for i in range(1, 6)).astype(float)
    bid_depth5 = sum(df[f"bid_s{i}"] for i in range(1, 6)).astype(float)
    f["ask_slope_5"] = (df["ask_p5"] - ask1).fillna(0) / ask_depth5 * 100  # $ per 100 shares
    f["bid_slope_5"] = (bid1 - df["bid_p5"]).fillna(0) / bid_depth5 * 100

    # ---------- C. depth and imbalance
    def imb(levels):
        b = sum(df[f"bid_s{i}"] for i in range(1, levels + 1)).astype(float)
        a = sum(df[f"ask_s{i}"] for i in range(1, levels + 1)).astype(float)
        return (b - a) / (b + a)

    f["imb_l1"] = imb(1)
    f["imb_l3"] = imb(3)
    f["imb_l5"] = imb(5)
    f["imb_l10"] = imb(10)
    f["log_bid_depth5"] = np.log1p(bid_depth5)
    f["log_ask_depth5"] = np.log1p(ask_depth5)
    f["log_l1_size_ratio"] = np.log1p(bq1) - np.log1p(aq1)

    # ---------- D. order flow imbalance
    ofi1 = _ofi_level(df, 1)
    ofi_ml = ofi1 + 0.5 * _ofi_level(df, 2) + 0.25 * _ofi_level(df, 3)  # depth-weighted multi-level
    depth1_avg = ((bq1 + aq1) / 2).rolling(50, min_periods=1).mean()
    f["ofi_10"] = _roll_sum(ofi1, 10) / depth1_avg
    f["ofi_50"] = _roll_sum(ofi1, 50) / depth1_avg
    f["ofi_200"] = _roll_sum(ofi1, 200) / depth1_avg
    f["mlofi_50"] = _roll_sum(ofi_ml, 50) / depth1_avg
    f["ofi_50_raw"] = _roll_sum(ofi1, 50)  # unnormalised, in shares

    # ---------- E. message flow
    typ, dirn, size = df["type"], df["direction"], df["size"].astype(float)
    is_trade = typ.isin([4, 5])
    # direction is the side of the resting order; aggressor sign is the opposite
    trade_sign = np.where(is_trade, -dirn, 0)
    signed_vol = pd.Series(trade_sign * size, index=df.index)
    tot_vol = pd.Series(np.where(is_trade, size, 0.0), index=df.index)
    f["trade_imb_50"] = _roll_sum(signed_vol, 50) / (_roll_sum(tot_vol, 50) + 1)
    f["trade_imb_200"] = _roll_sum(signed_vol, 200) / (_roll_sum(tot_vol, 200) + 1)
    sub = typ.eq(1)
    canc = typ.isin([2, 3])
    f["submit_imb_100"] = _roll_sum(pd.Series(np.where(sub, dirn, 0), index=df.index), 100) / (_roll_sum(sub.astype(float), 100) + 1)
    f["cancel_imb_100"] = _roll_sum(pd.Series(np.where(canc, dirn, 0), index=df.index), 100) / (_roll_sum(canc.astype(float), 100) + 1)
    f["cancel_submit_ratio_100"] = _roll_sum(canc.astype(float), 100) / (_roll_sum(sub.astype(float), 100) + 1)
    t = df["time"]
    f["event_rate_100"] = 100.0 / (t - t.shift(100)).clip(lower=1e-3).bfill()
    last_trade_t = t.where(is_trade).ffill()
    f["log_time_since_trade"] = np.log1p((t - last_trade_t).fillna(60.0))

    # ---------- F. returns and volatility (log-mid returns, in bps, over event windows)
    logm = np.log(mid)
    f["ret_10"] = (logm - logm.shift(10)).fillna(0) * 1e4
    f["ret_50"] = (logm - logm.shift(50)).fillna(0) * 1e4
    f["ret_200"] = (logm - logm.shift(200)).fillna(0) * 1e4
    f["rv_200"] = np.sqrt((logm.diff().fillna(0) * 1e4).pow(2).rolling(200, min_periods=1).sum())

    f = f.replace([np.inf, -np.inf], np.nan)
    assert f.shape[1] == 36, f.shape
    return f
