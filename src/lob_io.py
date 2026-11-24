"""Load LOBSTER message + orderbook files into one tidy DataFrame."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MSG_COLS = ["time", "type", "order_id", "size", "price", "direction"]
DUMMY_ASK, DUMMY_BID = 9_999_999_999, -9_999_999_999


def book_cols(levels: int) -> list[str]:
    cols = []
    for i in range(1, levels + 1):
        cols += [f"ask_p{i}", f"ask_s{i}", f"bid_p{i}", f"bid_s{i}"]
    return cols


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    # Official LOBSTER files have no header; some redistributed copies add one.
    first = path.open().readline()
    has_header = any(c.isalpha() for c in first)
    df = pd.read_csv(path, header=0 if has_header else None)
    df.columns = names[: df.shape[1]]
    return df


def find_files(data_dir: Path) -> tuple[Path, Path, int]:
    msg = sorted(data_dir.glob("*_message_*.csv"))
    ob = sorted(data_dir.glob("*_orderbook_*.csv"))
    if not msg or not ob:
        raise FileNotFoundError(f"no LOBSTER message/orderbook csv pair in {data_dir}")
    levels = int(ob[0].stem.split("_")[-1])
    return msg[0], ob[0], levels


def load_lobster(data_dir: Path, levels_used: int = 10) -> pd.DataFrame:
    """One row per message, with the book state *after* that message."""
    mpath, bpath, levels = find_files(Path(data_dir))
    msg = _read(mpath, MSG_COLS)
    book = _read(bpath, book_cols(levels))
    if len(msg) != len(book):
        raise ValueError("message and orderbook files have different lengths")
    levels_used = min(levels_used, levels)
    df = pd.concat([msg, book[book_cols(levels_used)]], axis=1)

    # LOBSTER pads missing levels with dummy prices; turn them into NaN / 0 size
    for i in range(1, levels_used + 1):
        df.loc[df[f"ask_p{i}"] >= DUMMY_ASK, [f"ask_p{i}", f"ask_s{i}"]] = [np.nan, 0]
        df.loc[df[f"bid_p{i}"] <= DUMMY_BID, [f"bid_p{i}", f"bid_s{i}"]] = [np.nan, 0]

    # Trading halts (type 7) and cross trades (type 6) are not book events
    df = df[df["type"].isin([1, 2, 3, 4, 5])].reset_index(drop=True)
    # Drop rows where either side of the book is empty (no mid-price defined)
    df = df.dropna(subset=["ask_p1", "bid_p1"]).reset_index(drop=True)

    # Prices: LOBSTER integer units of $0.0001 -> dollars
    price_cols = ["price"] + [c for c in df.columns if c.startswith(("ask_p", "bid_p"))]
    df[price_cols] = df[price_cols] / 10_000.0
    df.attrs["source"] = mpath.name
    return df
