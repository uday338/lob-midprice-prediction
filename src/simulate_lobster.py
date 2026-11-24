"""
Agent-based limit order book simulator that writes LOBSTER-format files.

Why this exists
---------------
The real LOBSTER sample files (https://lobsterdata.com/info/DataSamples.php) could
not be downloaded from the build environment, so this script generates one trading
day of NASDAQ-style data with the exact same file layout:

  <TICKER>_<DATE>_34200000_57600000_message_10.csv    (time, type, order_id, size, price, direction)
  <TICKER>_<DATE>_34200000_57600000_orderbook_10.csv  (ask_p1, ask_s1, bid_p1, bid_s1, ... level 10)

Prices are integers in units of $0.0001 (LOBSTER convention), tick = $0.01 = 100.
Message types: 1 submit, 2 partial cancel, 3 full delete, 4 visible execution,
5 hidden execution. Direction: 1 = buy limit order (bid side), -1 = sell limit order.
For executions the direction is the side of the *resting* order that was hit,
so direction=-1 on a type-4 row means a buyer-initiated trade (an ask was hit).

The model (simple, but with the right moving parts)
--------------------------------------------------
* Event arrivals follow a Poisson process with an intraday U-shaped intensity.
* Each event is a limit submission, a cancellation, or a marketable order.
* A latent "informed flow" signal s_t (an Ornstein-Uhlenbeck process in event time)
  tilts the buy/sell mix of market orders, where new limit orders are placed and
  which side cancels. Because s_t is persistent, recent order-flow imbalance carries
  information about the next mid-price moves. That is the effect real microstructure
  papers find (Cont, Kukanov & Stoikov 2014), and it is what the model has to learn.
* s_t is partly anchored to a slowly drifting "fundamental" value, so the absolute
  price level moves around during the day. This is what makes price-level features
  look useful in-sample and break out-of-sample.

Swap in real data: everything downstream reads the LOBSTER layout, so running
`python src/run_pipeline.py --data-dir <folder with real LOBSTER csvs>` works unchanged.
"""
from __future__ import annotations

import argparse
import bisect
from collections import deque
from pathlib import Path

import numpy as np

TICK = 100  # $0.01 in LOBSTER price units
OPEN_S, CLOSE_S = 34200.0, 57600.0  # 09:30 to 16:00 in seconds after midnight
LEVELS = 10


class Side:
    """One side of the book: sorted price levels, each a FIFO queue of [order_id, size]."""

    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self.prices: list[int] = []  # ascending
        self.queues: dict[int, deque] = {}
        self.volume: dict[int, int] = {}

    def best(self) -> int | None:
        if not self.prices:
            return None
        return self.prices[-1] if self.is_bid else self.prices[0]

    def add(self, price: int, oid: int, size: int):
        if price not in self.queues:
            bisect.insort(self.prices, price)
            self.queues[price] = deque()
            self.volume[price] = 0
        self.queues[price].append([oid, size])
        self.volume[price] += size

    def remove_level_if_empty(self, price: int):
        if self.volume.get(price, 1) <= 0:
            i = bisect.bisect_left(self.prices, price)
            del self.prices[i]
            del self.queues[price]
            del self.volume[price]

    def top(self, n: int) -> list[tuple[int, int]]:
        ps = self.prices[::-1][:n] if self.is_bid else self.prices[:n]
        return [(p, self.volume[p]) for p in ps]

    def n_levels(self) -> int:
        return len(self.prices)


def simulate(n_events: int = 400_000, seed: int = 7, start_price: float = 585.0):
    rng = np.random.default_rng(seed)
    bids, asks = Side(True), Side(False)
    oid_counter = [10_000_000]
    live: dict[int, tuple[int, int]] = {}  # oid -> (direction, price)

    def new_oid():
        oid_counter[0] += 1
        return oid_counter[0]

    def lot():
        # mostly round lots of 100, some odd lots and some large orders
        u = rng.random()
        if u < 0.70:
            return 100
        if u < 0.82:
            return int(rng.integers(1, 100))
        if u < 0.95:
            return 100 * int(rng.integers(2, 6))
        return 100 * int(rng.integers(6, 30))

    # initial book: sparse levels like real AAPL in 2012 (gaps of several ticks)
    p0 = int(round(start_price * 100)) * TICK
    for side, sign in ((bids, -1), (asks, 1)):
        px = p0 + sign * 5 * TICK
        for _ in range(40):
            for _ in range(int(rng.integers(1, 4))):
                oid = new_oid()
                side.add(px, oid, lot())
                live[oid] = (1 if side.is_bid else -1, px)
            px += sign * TICK * int(rng.integers(1, 6))

    msgs = np.zeros((n_events, 6), dtype=np.float64)
    book = np.zeros((n_events, 4 * LEVELS), dtype=np.int64)
    row = 0

    def snapshot(t, typ, oid, size, price, direction):
        nonlocal row
        if row >= n_events:
            return
        msgs[row] = (t, typ, oid, size, price, direction)
        a, b = asks.top(LEVELS), bids.top(LEVELS)
        r = book[row]
        for lvl in range(LEVELS):
            # LOBSTER pads empty levels with dummy prices (9999999999 / -9999999999) and size 0
            ap, asz = a[lvl] if lvl < len(a) else (9_999_999_999, 0)
            bp, bsz = b[lvl] if lvl < len(b) else (-9_999_999_999, 0)
            r[4 * lvl: 4 * lvl + 4] = (ap, asz, bp, bsz)
        row += 1

    # latent processes
    s = 0.0  # informed flow signal, roughly in [-1, 1]
    fundamental = float(p0)
    t = OPEN_S
    avg_rate = n_events / (CLOSE_S - OPEN_S)

    while row < n_events:
        # --- clock: U-shaped intraday intensity
        frac = (t - OPEN_S) / (CLOSE_S - OPEN_S)
        rate = avg_rate * (0.65 + 1.4 * (frac - 0.5) ** 2 * 4 * 0.5)
        # bursts: activity clusters (crude self-excitation)
        rate *= 1.0 + 2.5 * abs(s)
        t += rng.exponential(1.0 / rate)
        t = min(t, CLOSE_S - 1e-6)

        # --- latent dynamics (event time)
        fundamental += rng.normal(0, 0.9 * TICK)
        mid = 0.5 * (bids.best() + asks.best())
        anchor = np.tanh((fundamental - mid) / (40 * TICK))
        s += -s / 600.0 + 0.004 * anchor + rng.normal(0, 0.035)
        s = float(np.clip(s, -1.0, 1.0))

        bb, ba = bids.best(), asks.best()
        spread_ticks = (ba - bb) // TICK

        # occasional hidden execution (type 5): no visible book change
        if rng.random() < 0.012:
            direction = 1 if rng.random() < 0.5 - 0.3 * s else -1
            px = bb if direction == 1 else ba
            snapshot(t, 5, 0, lot(), px, direction)
            continue

        u = rng.random()
        p_market = 0.022 * (1 + 1.5 * abs(s))
        p_cancel = 0.47
        if u < p_market:
            # ---- marketable order; buy prob tilted by the informed signal
            buy = rng.random() < 0.5 + 0.25 * s
            size = lot() if rng.random() < 0.8 else lot() * int(rng.integers(2, 5))
            side = asks if buy else bids
            resting_dir = -1 if buy else 1
            while size > 0 and side.n_levels() > 3 and row < n_events:
                px = side.best()
                q = side.queues[px]
                o = q[0]
                fill = min(size, o[1])
                o[1] -= fill
                side.volume[px] -= fill
                size -= fill
                done_oid = o[0]
                if o[1] == 0:
                    q.popleft()
                    live.pop(done_oid, None)
                side.remove_level_if_empty(px)
                snapshot(t, 4, done_oid, fill, px, resting_dir)
                t += 1e-6
        elif u < p_market + p_cancel:
            # ---- cancellation; side tilted against the informed direction
            ask_side = rng.random() < 0.5 + 0.10 * s
            side = asks if ask_side else bids
            if side.n_levels() <= 5:
                continue  # never let a side run empty
            # pick a level with prob ~ volume, biased toward the touch less than deep book
            ps = side.prices
            if not ps:
                continue
            k = len(ps)
            # distance from touch
            idx_from_touch = int(min(k - 1, rng.geometric(0.18) - 1))
            px = ps[-1 - idx_from_touch] if side.is_bid else ps[idx_from_touch]
            q = side.queues[px]
            j = int(rng.integers(0, len(q)))
            o = q[j]
            direction = 1 if side.is_bid else -1
            if rng.random() < 0.08 and o[1] > 1:
                cut = int(rng.integers(1, o[1]))
                o[1] -= cut
                side.volume[px] -= cut
                snapshot(t, 2, o[0], cut, px, direction)
            else:
                del q[j]
                side.volume[px] -= o[1]
                live.pop(o[0], None)
                side.remove_level_if_empty(px)
                snapshot(t, 3, o[0], o[1], px, direction)
        else:
            # ---- new limit order; side tilted toward the informed direction
            is_bid = rng.random() < 0.5 + 0.12 * s
            side = bids if is_bid else asks
            # keep both sides deep enough
            if bids.n_levels() < 12:
                side, is_bid = bids, True
            elif asks.n_levels() < 12:
                side, is_bid = asks, False
            best_same = side.best()
            if spread_ticks > 1 and rng.random() < 0.10 * (1 + 0.6 * (s if is_bid else -s)):
                # price improvement inside the spread
                step = int(rng.integers(1, spread_ticks))
                px = bb + step * TICK if is_bid else ba - step * TICK
            else:
                d = int(rng.geometric(0.06)) - 1  # ticks behind the touch
                px = best_same - d * TICK if is_bid else best_same + d * TICK
            oid = new_oid()
            size = lot()
            side.add(px, oid, size)
            live[oid] = (1 if is_bid else -1, px)
            snapshot(t, 1, oid, size, px, 1 if is_bid else -1)

    msgs, book = msgs[:row], book[:row]
    # stretch the clock so the session spans exactly 09:30-16:00
    span = msgs[-1, 0] - OPEN_S
    msgs[:, 0] = OPEN_S + (msgs[:, 0] - OPEN_S) * ((CLOSE_S - OPEN_S - 1.0) / span)
    return msgs, book


def write_lobster(msgs, book, out_dir: Path, ticker="SIMX", date="2012-06-21"):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{ticker}_{date}_34200000_57600000"
    mpath = out_dir / f"{base}_message_{LEVELS}.csv"
    bpath = out_dir / f"{base}_orderbook_{LEVELS}.csv"
    # LOBSTER files have no header row
    np.savetxt(mpath, msgs, delimiter=",", fmt=["%.9f", "%d", "%d", "%d", "%d", "%d"])
    np.savetxt(bpath, book, delimiter=",", fmt="%d")
    return mpath, bpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=Path("data/simulated"))
    args = ap.parse_args()
    msgs, book = simulate(args.n_events, args.seed)
    m, b = write_lobster(msgs, book, args.out_dir)
    print(f"wrote {len(msgs):,} events\n  {m}\n  {b}")


if __name__ == "__main__":
    main()
