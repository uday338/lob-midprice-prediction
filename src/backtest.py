"""
Simple event-time backtest of the classifier's signal.

Rule: at event t, score = P(up) - P(down). If score > theta go long 1 share, if
score < -theta go short 1 share, and close the position exactly k events later.
Every event where the rule fires counts as one trade, so positions overlap. This
measures the quality of the *signal*, not a deployable strategy (no inventory limits,
no latency, no queue position, no market impact).

Per-trade P&L in dollars per share:
  raw (mid-to-mid)  = side * (mid[t+k] - mid[t])
  net of spread     = raw - (spread[t] / 2 + spread[t+k] / 2)
The cost term is what you pay to cross the spread on entry and on exit with
marketable orders, measured against the mid.

Sharpe here is the per-trade Sharpe, mean(pnl) / std(pnl). With overlapping trades
from a single day, an annualised Sharpe would be misleading, so it is not reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(proba: np.ndarray, mid: np.ndarray, spread: np.ndarray, idx: np.ndarray, k: int, theta: float) -> dict:
    score = proba[:, 2] - proba[:, 0]
    side = np.where(score > theta, 1, np.where(score < -theta, -1, 0))
    entry = idx[side != 0]
    s = side[side != 0]
    exit_ = entry + k
    keep = exit_ < len(mid)
    entry, exit_, s = entry[keep], exit_[keep], s[keep]
    raw = s * (mid[exit_] - mid[entry])
    cost = spread[entry] / 2 + spread[exit_] / 2
    net = raw - cost
    n = len(raw)
    if n < 2:
        return dict(theta=theta, n_trades=n)
    return dict(
        theta=float(theta),
        n_trades=int(n),
        hit_rate=float((raw > 0).mean()),
        mean_raw_pnl=float(raw.mean()),
        mean_cost=float(cost.mean()),
        mean_net_pnl=float(net.mean()),
        sharpe_raw=float(raw.mean() / raw.std(ddof=1)),
        sharpe_net=float(net.mean() / net.std(ddof=1)),
        total_raw_pnl=float(raw.sum()),
        total_net_pnl=float(net.sum()),
        breakeven_cost=float(raw.mean()),
    )


def equity_curves(proba, mid, spread, idx, k, theta) -> pd.DataFrame:
    score = proba[:, 2] - proba[:, 0]
    side = np.where(score > theta, 1, np.where(score < -theta, -1, 0))
    m = side != 0
    entry, s = idx[m], side[m]
    keep = entry + k < len(mid)
    entry, s = entry[keep], s[keep]
    raw = s * (mid[entry + k] - mid[entry])
    cost = spread[entry] / 2 + spread[entry + k] / 2
    return pd.DataFrame({"event": entry, "raw": raw.cumsum(), "net": (raw - cost).cumsum()})


def run_backtest_non_overlapping(proba, mid, spread, idx, k, theta) -> dict:
    """Same rule, but hold at most one position: after entering, ignore signals until the exit."""
    score = proba[:, 2] - proba[:, 0]
    keep_rows, j = [], 0
    while j < len(idx):
        if abs(score[j]) > theta and idx[j] + k < len(mid):
            keep_rows.append(j)
            j += k
        else:
            j += 1
    keep_rows = np.array(keep_rows, dtype=int)
    if len(keep_rows) == 0:
        return dict(theta=float(theta), n_trades=0)
    return run_backtest(proba[keep_rows], mid, spread, idx[keep_rows], k, theta)
