"""LightGBM training, baselines, and importance diagnostics."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

LGB_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=500,  # large leaves: LOB labels are noisy, small leaves memorise noise
    feature_fraction=0.8,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l2=5.0,
    n_estimators=2000,
    verbose=-1,
    n_jobs=4,
)


def train_lgbm(Xtr, ytr, Xva, yva, seed=0, params=None):
    p = dict(LGB_PARAMS, random_state=seed, **(params or {}))
    model = lgb.LGBMClassifier(**p)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def macro_f1(y, yhat) -> float:
    return float(f1_score(y, yhat, average="macro"))


def random_baseline_f1(y_true, train_prior, n_draws=200, seed=0) -> tuple[float, float]:
    """Macro F1 of a classifier that guesses at random with the training class frequencies."""
    rng = np.random.default_rng(seed)
    scores = [
        macro_f1(y_true, rng.choice(3, size=len(y_true), p=train_prior))
        for _ in range(n_draws)
    ]
    return float(np.mean(scores)), float(np.std(scores))


def gain_importance(model, cols) -> pd.Series:
    imp = model.booster_.feature_importance(importance_type="gain")
    return pd.Series(imp / imp.sum(), index=cols).sort_values(ascending=False)


def perm_importance(model, X, y, seed=0, n_repeats=3, max_rows=20000) -> pd.Series:
    """Drop in macro F1 when one column is shuffled. Computed on held-out data."""
    if len(X) > max_rows:
        X, y = X.iloc[-max_rows:], y[-max_rows:]
    r = permutation_importance(model, X, y, scoring="f1_macro", n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def adversarial_auc(X_a: pd.DataFrame, X_b: pd.DataFrame, seed=0) -> pd.Series:
    """Per-feature AUC for telling period A rows from period B rows.

    An AUC near 1.0 (or 0.0) means the feature's distribution has shifted completely
    between periods, so whatever the model learned from it in A cannot transfer to B.
    """
    out = {}
    y = np.r_[np.zeros(len(X_a)), np.ones(len(X_b))]
    for c in X_a.columns:
        x = np.r_[X_a[c].to_numpy(), X_b[c].to_numpy()]
        ok = ~np.isnan(x)
        auc = roc_auc_score(y[ok], x[ok])
        out[c] = max(auc, 1 - auc)
    return pd.Series(out).sort_values(ascending=False)
