"""
End-to-end pipeline:
  load LOBSTER -> 36 features -> 3-class labels -> 70/15/15 time split ->
  LightGBM (all features) -> importance + leakage diagnosis ->
  LightGBM (no absolute-price features) -> backtest -> results/

Usage:
  python src/simulate_lobster.py                      # writes data/simulated/
  python src/run_pipeline.py                          # uses data/simulated/
  python src/run_pipeline.py --data-dir data/lobster  # real LOBSTER files
"""
from __future__ import annotations

import argparse
import warnings
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from backtest import equity_curves, run_backtest, run_backtest_non_overlapping  # noqa: E402
from features import ABSOLUTE_PRICE_FEATURES, build_features  # noqa: E402
from labels import CLASS_NAMES, choose_alpha, make_labels, smoothed_future_return, time_split  # noqa: E402
from lob_io import load_lobster  # noqa: E402
from model import (adversarial_auc, gain_importance, macro_f1, perm_importance,  # noqa: E402
                   random_baseline_f1, train_lgbm)

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "simulated")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--horizon", type=int, default=50, help="label horizon k, in events")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, figs = args.out_dir, args.out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    k = args.horizon

    # ---------------- data
    df = load_lobster(args.data_dir)
    log(f"loaded {len(df):,} events from {df.attrs['source']}")
    X = build_features(df)
    mid = ((df["ask_p1"] + df["bid_p1"]) / 2).to_numpy()
    spread = (df["ask_p1"] - df["bid_p1"]).to_numpy()

    # ---------------- labels + split
    tr, va, te = time_split(len(df), k)
    fret = smoothed_future_return(pd.Series(mid), k)
    alpha = choose_alpha(fret.iloc[tr])
    y = make_labels(fret, alpha).to_numpy()
    for name, ix in (("train", tr), ("val", va), ("test", te)):
        assert not np.isnan(y[ix]).any(), name
    y = np.nan_to_num(y, nan=1).astype(int)
    prior = np.bincount(y[tr], minlength=3) / len(tr)
    dist = {n: (np.bincount(y[ix], minlength=3) / len(ix)).round(3).tolist() for n, ix in (("train", tr), ("val", va), ("test", te))}
    log(f"horizon k={k} events, alpha={alpha:.2e}; class shares (down, flat, up): {dist}")

    results = {
        "data_source": df.attrs["source"],
        "n_events": int(len(df)),
        "n_features": int(X.shape[1]),
        "horizon_events": k,
        "alpha": alpha,
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "class_shares": dist,
        "session_minutes_per_split": {
            n: float((df["time"].iloc[ix[-1]] - df["time"].iloc[ix[0]]) / 60) for n, ix in (("train", tr), ("val", va), ("test", te))
        },
    }

    # ---------------- baselines
    rb_val = random_baseline_f1(y[va], prior, seed=args.seed)
    rb_test = random_baseline_f1(y[te], prior, seed=args.seed)
    results["random_baseline_macro_f1"] = {"val": rb_val[0], "test": rb_test[0], "test_std": rb_test[1]}
    log(f"random baseline macro F1: val {rb_val[0]:.4f}  test {rb_test[0]:.4f} (+/- {rb_test[1]:.4f})")

    # ---------------- model A: all 36 features
    cols_all = list(X.columns)
    mA = train_lgbm(X.iloc[tr], y[tr], X.iloc[va], y[va], seed=args.seed)
    fA = {s: macro_f1(y[ix], mA.predict(X.iloc[ix])) for s, ix in (("train", tr), ("val", va), ("test", te))}
    log(f"model A (all {len(cols_all)}): {fA}  trees={mA.best_iteration_}")
    gA = gain_importance(mA, cols_all)
    pA = perm_importance(mA, X.iloc[te], y[te], seed=args.seed)

    # leakage diagnostics
    adv = adversarial_auc(X.iloc[tr], X.iloc[te])
    results["model_all_features"] = {
        "macro_f1": fA,
        "best_iteration": int(mA.best_iteration_),
        "gain_importance_top10": gA.head(10).round(4).to_dict(),
        "abs_price_gain_share": float(gA[ABSOLUTE_PRICE_FEATURES].sum()),
        "abs_price_gain_ranks": {c: int(list(gA.index).index(c)) + 1 for c in ABSOLUTE_PRICE_FEATURES},
        "test_permutation_importance_abs_price": pA[ABSOLUTE_PRICE_FEATURES].round(5).to_dict(),
    }
    results["train_vs_test_shift_auc"] = adv.head(10).round(3).to_dict()
    price_range = {
        n: [float(mid[ix].min()), float(mid[ix].max())] for n, ix in (("train", tr), ("val", va), ("test", te))
    }
    results["mid_price_range"] = price_range
    log(f"abs-price share of gain: {results['model_all_features']['abs_price_gain_share']:.3f}; mid ranges {price_range}")

    # ---------------- model B: drop absolute-price features
    cols_rel = [c for c in cols_all if c not in ABSOLUTE_PRICE_FEATURES]
    mB = train_lgbm(X.iloc[tr][cols_rel], y[tr], X.iloc[va][cols_rel], y[va], seed=args.seed)
    fB = {s: macro_f1(y[ix], mB.predict(X.iloc[ix][cols_rel])) for s, ix in (("train", tr), ("val", va), ("test", te))}
    log(f"model B (no abs price, {len(cols_rel)}): {fB}  trees={mB.best_iteration_}")
    gB = gain_importance(mB, cols_rel)
    pB = perm_importance(mB, X.iloc[te][cols_rel], y[te], seed=args.seed)

    # seed robustness for the final model
    seed_f1 = []
    for s in range(1, 4):
        m = train_lgbm(X.iloc[tr][cols_rel], y[tr], X.iloc[va][cols_rel], y[va], seed=s)
        seed_f1.append(macro_f1(y[te], m.predict(X.iloc[te][cols_rel])))
    yhatB = mB.predict(X.iloc[te][cols_rel])
    results["model_no_abs_price"] = {
        "n_features": len(cols_rel),
        "macro_f1": fB,
        "best_iteration": int(mB.best_iteration_),
        "test_macro_f1_other_seeds": seed_f1,
        "lift_over_random_test_pct": 100 * (fB["test"] / rb_test[0] - 1),
        "gain_importance_top10": gB.head(10).round(4).to_dict(),
        "test_permutation_importance_top10": pB.head(10).round(5).to_dict(),
        "test_confusion_matrix": confusion_matrix(y[te], yhatB).tolist(),
        "test_report": classification_report(y[te], yhatB, target_names=CLASS_NAMES, output_dict=True),
    }
    log(f"lift over random (test): {results['model_no_abs_price']['lift_over_random_test_pct']:.1f}%  other seeds {np.round(seed_f1, 4)}")

    # ---------------- backtest: choose theta on validation, report on test
    pva = mB.predict_proba(X.iloc[va][cols_rel])
    pte = mB.predict_proba(X.iloc[te][cols_rel])
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    val_bt = [run_backtest(pva, mid, spread, va, k, th) for th in grid]
    test_bt = [run_backtest(pte, mid, spread, te, k, th) for th in grid]
    ok = [b for b in val_bt if b.get("n_trades", 0) >= 500]
    best = max(ok, key=lambda b: b["sharpe_raw"])
    theta = best["theta"]
    bt = run_backtest(pte, mid, spread, te, k, theta)
    bt_non_overlap = run_backtest_non_overlapping(pte, mid, spread, te, k, theta)
    results["backtest"] = {
        "theta_chosen_on_val": theta,
        "test": bt,
        "test_non_overlapping": bt_non_overlap,
        "theta_grid_val": val_bt,
        "theta_grid_test": test_bt,
        "avg_spread_test": float(spread[te].mean()),
    }
    log(f"backtest theta={theta}: {json.dumps({k_: round(v, 4) if isinstance(v, float) else v for k_, v in bt.items()})}")

    # ---------------- figures
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(9, 3.2))
    t_h = (df["time"].to_numpy() - 34200) / 3600 + 9.5
    ax.plot(t_h, mid, lw=0.6, color="#3b6ea5")
    for ix, c, n in ((tr, "#3b6ea5", "train"), (va, "#e0a030", "val"), (te, "#c0504d", "test")):
        ax.axvspan(t_h[ix[0]], t_h[ix[-1]], color=c, alpha=0.08)
        ax.text(t_h[ix[0]] + 0.05, mid.max(), n, va="top", fontsize=9)
    ax.set(xlabel="hour of day", ylabel="mid price ($)", title="Mid price and the 70/15/15 time split")
    fig.tight_layout(); fig.savefig(figs / "mid_price_split.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, g, title in ((axes[0], gA, "All 36 features"), (axes[1], gB, "Absolute-price features removed")):
        top = g.head(15)[::-1]
        colors = ["#c0504d" if c in ABSOLUTE_PRICE_FEATURES else "#3b6ea5" for c in top.index]
        ax.barh(top.index, top.values, color=colors)
        ax.set(title=title, xlabel="share of total gain")
    fig.suptitle("LightGBM gain importance (red = absolute price level)")
    fig.tight_layout(); fig.savefig(figs / "feature_importance.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    top = adv.head(12)[::-1]
    ax.barh(top.index, top.values, color=["#c0504d" if c in ABSOLUTE_PRICE_FEATURES else "#3b6ea5" for c in top.index])
    ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    ax.set(xlim=(0.4, 1.0), xlabel="AUC for telling train rows from test rows", title="Distribution shift by feature (train vs test)")
    fig.tight_layout(); fig.savefig(figs / "train_test_shift.png"); plt.close(fig)

    eq = equity_curves(pte, mid, spread, te, k, theta)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(eq["event"], eq["raw"], label="raw (mid to mid)", color="#3b6ea5")
    ax.plot(eq["event"], eq["net"], label="net of spread cost", color="#c0504d")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="event index (test period)", ylabel="cumulative P&L ($ per share)", title=f"Test-period backtest, theta={theta}")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figs / "backtest_equity.png"); plt.close(fig)

    def _jsonable(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=_jsonable))
    gA.rename("gain_share").to_csv(out / "importance_all_features.csv")
    gB.rename("gain_share").to_csv(out / "importance_no_abs_price.csv")
    log(f"wrote {out / 'metrics.json'} and figures/")


if __name__ == "__main__":
    main()
