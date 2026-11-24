# Study Guide: Limit Order Book Mid-Price Prediction

This guide explains every idea used in the project, in the order you would explain it in an interview. Each section ends with the points worth remembering. Section 10 has interview questions with model answers.

Contents
1. The project in 30 seconds
2. Limit order book basics
3. The LOBSTER data format
4. The 36 features
5. Labels
6. Time-series splits and leakage
7. LightGBM
8. Metrics
9. Backtesting
10. Interview questions and answers
11. Further reading

---

## 1. The project in 30 seconds

> "I built a classifier that predicts whether a stock's mid-price will go up, down, or stay flat over the next 50 order-book events. I engineered 36 microstructure features from the order book and the message stream: order flow imbalance, microprice, depth imbalance, trade and cancel flow. I used a chronological 70/15/15 split with purging, trained LightGBM, and beat a random baseline on macro F1. Importance analysis showed absolute-price features were picking up the day's price level rather than anything that generalises, so I removed them. Then I backtested the signal. It had a positive raw Sharpe, but it did not cover the cost of crossing the spread, which is the usual result for short-horizon signals traded with market orders."

Be ready to say which data you used. In this repo it is a simulator that writes LOBSTER-format files, because the real files could not be downloaded in the build environment. Rerun on real LOBSTER data before quoting numbers as "NASDAQ".

---

## 2. Limit order book basics

**Limit order.** "Buy 100 shares at $585.10 or better." If it cannot execute right away, it rests in the book and waits. Resting limit orders are *liquidity*.

**Market order (or marketable limit order).** "Buy 100 shares now at the best available price." It executes against resting orders on the other side. It *takes* liquidity.

**Cancellation.** A resting order is removed (fully: a delete; partly: a partial cancel). On NASDAQ, most limit orders are cancelled before they ever trade. In the AAPL sample, deletes are about 40% of all messages and executions under 10%.

**The book.** All resting orders, grouped by price:
* **Bid side**: buy orders. The highest bid is the **best bid** (level 1). The next distinct price below is level 2, and so on.
* **Ask side** (or offer side): sell orders. The lowest ask is the **best ask**.
* Each level has a price and a total size (shares) called the **queue** at that level.

```
        price    size
ask L3  585.20    300
ask L2  585.15    100
ask L1  585.12    200   <- best ask
--------------------------------  spread = 585.12 - 585.05 = $0.07
bid L1  585.05    500   <- best bid
bid L2  585.03    100
bid L3  585.00   1200
```

**Spread** = best ask − best bid. It is the round-trip cost of trading immediately: buy at the ask, sell at the bid.

**Mid-price** = (best bid + best ask) / 2. Not a price anyone trades at, but the standard reference for "where the price is". The project predicts its direction.

**Tick size.** The minimum price step: $0.01 for US stocks above $1. A stock whose spread is almost always one tick is **large-tick** (e.g. a $20 stock). A stock with a spread of many ticks is **small-tick** (AAPL at $585 in 2012 had spreads of 10 to 30 ticks). For large-tick stocks the mid moves rarely and queue sizes carry most of the information. For small-tick stocks the mid moves often and spread dynamics matter.

**Price-time priority.** At a given price, orders execute first in, first out. Your place in the queue decides whether your passive order fills.

**Hidden orders.** Orders not shown in the book. When they execute you see a "hidden execution" message (type 5) with no visible book change.

**Event time vs clock time.** Sampling every N events (as this project does) adapts to activity: busy periods give more samples. Sampling every N seconds gives a fixed rate but mixes quiet and busy periods. Both are common.

**Why the mid moves.** The mid changes only when the best bid or best ask changes. That happens when (a) a level-1 queue is fully eaten by market orders, (b) the whole level-1 queue is cancelled, or (c) a new order improves the price inside the spread. So the size of the level-1 queues and the flow that adds to or removes from them drive short-term moves. That is why imbalance and order flow features work.

Remember: bid, ask, spread, mid, levels, queues, ticks; market orders take liquidity, limit orders provide it; the mid moves when a level-1 queue empties or the spread is improved.

---

## 3. The LOBSTER data format

LOBSTER (lobsterdata.com) rebuilds the NASDAQ order book from the exchange's message feed (ITCH). For each stock and day it gives two CSV files with the same number of rows:

**Message file**: one row per event.

| Column | Meaning |
|---|---|
| time | seconds after midnight, nanosecond precision (34200 = 09:30) |
| type | 1 new limit order, 2 partial cancel, 3 full delete, 4 visible execution, 5 hidden execution, 6 cross trade, 7 trading halt |
| order_id | exchange order reference |
| size | shares |
| price | dollars × 10,000 (so 5851200 = $585.12) |
| direction | 1 = buy limit order, −1 = sell limit order. For executions this is the side of the *resting* order, so −1 on a type-4 row means a buyer hit the ask |

**Orderbook file**: the book right *after* that event: ask price 1, ask size 1, bid price 1, bid size 1, ask price 2, ... up to the chosen number of levels. Empty levels hold dummy prices (±9999999999).

Two traps:
* The trade direction is inverted. Aggressor sign = −direction.
* Each row's book is *after* the message, so a feature at row t can use row t's book. The label must only use rows after t.

FI-2010 is the other common benchmark (Finnish stocks, 10 days, already normalised features and labels). It is convenient but you cannot build your own features from raw messages with it, which is why LOBSTER fits this project better.

---

## 4. The 36 features

All features at row t use only the book after event t and messages up to t. Windows are counted in events. Notation: `b1, a1` best bid and ask; `qb1, qa1` their sizes; `m` mid; `s` spread.

### A. Absolute price level (5): the leaky group

| Feature | Formula |
|---|---|
| mid_price | (a1 + b1) / 2 |
| best_bid | b1 |
| best_ask | a1 |
| microprice | (a1·qb1 + b1·qa1) / (qb1 + qa1) |
| book_vwap_5 | volume-weighted average price of the top 5 levels on both sides |

These say *where* the price is, not *what it is about to do*. A stock's price level is non-stationary: it wanders and never revisits the same range on schedule. A tree model can still split on it ("when price was between 586 and 587 it tended to go up") and that pattern fits the training period but is noise in the future. That is the "price-level leakage" in the project. Strictly it is spurious correlation plus distribution shift, not leakage of future information, but "leakage" is the common word for it.

### B. Spread and book shape (8)

| Feature | Formula | Intuition |
|---|---|---|
| spread | a1 − b1 | Wide spread means the mid can jump further when the spread closes |
| spread_bps | s / m × 10,000 | Same, scale-free |
| microprice_dev | (microprice − m) / s | Where the "fair" price sits inside the spread, between −0.5 and 0.5. Positive = bid queue bigger than ask queue, so the price leans up |
| wmid5_dev | (book_vwap_5 − m) / s | Same idea using 5 levels |
| ask_gap_12, bid_gap_12 | a2 − a1, b1 − b2 | If level 1 empties, how far does the price jump? |
| ask_slope_5, bid_slope_5 | (distance from level 1 to level 5) / (depth on 5 levels) × 100 | How "thin" the book is: dollars of price movement per 100 shares of depth |

**Microprice.** The mid ignores queue sizes. If 900 shares are bid and 100 are offered, the next move is more likely up, because the ask queue is the one about to run out. Weighting each side's price by the *opposite* side's size moves the estimate toward the thin side. Stoikov (2018) shows a refined version is a better short-term predictor of future mids than the mid itself.

### C. Depth and imbalance (7)

| Feature | Formula |
|---|---|
| imb_l1, imb_l3, imb_l5, imb_l10 | (bid depth − ask depth) / (bid depth + ask depth) over the first 1, 3, 5, 10 levels |
| log_bid_depth5, log_ask_depth5 | log(1 + total size on 5 levels) |
| log_l1_size_ratio | log(1 + qb1) − log(1 + qa1) |

Queue imbalance is one of the most robust short-horizon predictors in the literature (Gould & Bonart 2016). Deeper levels add less but help when level 1 is thin.

### D. Order flow imbalance (5)

**OFI (Cont, Kukanov & Stoikov 2014).** Imbalance looks at the *state* of the book. OFI looks at *changes*: how much buying pressure (bids added, asks removed) minus selling pressure arrived since the last event. For each event n:

```
e_n =  qb_n·1{b_n ≥ b_{n-1}}  − qb_{n-1}·1{b_n ≤ b_{n-1}}
     − qa_n·1{a_n ≤ a_{n-1}}  + qa_{n-1}·1{a_n ≥ a_{n-1}}
```

Read it case by case:
* bid price unchanged: contributes qb_n − qb_{n−1}, the change in bid queue (added minus cancelled or executed);
* bid price goes up: + the whole new bid queue (new buying interest at a better price);
* bid price goes down: − the whole old bid queue (it was wiped out);
* the ask side is the mirror image with opposite sign.

Summing e_n over a window gives OFI. Cont et al. find that mid-price changes are close to linear in OFI, with a slope inversely proportional to depth. That is why the project divides by average level-1 depth.

| Feature | Definition |
|---|---|
| ofi_10, ofi_50, ofi_200 | Level-1 OFI summed over the last 10, 50, 200 events, divided by average level-1 depth |
| mlofi_50 | Multi-level OFI: level 1 + 0.5 × level 2 + 0.25 × level 3, 50 events |
| ofi_50_raw | Level-1 OFI over 50 events in shares, unnormalised |

### E. Message flow (7)

| Feature | Definition | Intuition |
|---|---|---|
| trade_imb_50, trade_imb_200 | (buyer-initiated − seller-initiated volume) / total volume over the window | Aggressive buying tends to continue (order splitting) and has price impact |
| submit_imb_100 | net sign of new limit orders (bid = +1, ask = −1) per submit, last 100 events | Where new liquidity is arriving |
| cancel_imb_100 | net sign of cancellations | Cancelling asks (or adding bids) is a bullish tell; cancellations often precede moves |
| cancel_submit_ratio_100 | cancels / submits | Liquidity being withdrawn |
| event_rate_100 | 100 / seconds taken by the last 100 events | Activity; moves cluster in busy periods |
| log_time_since_trade | log(1 + seconds since the last execution) | Calm vs active market |

### F. Returns and volatility (4)

| Feature | Definition |
|---|---|
| ret_10, ret_50, ret_200 | log mid return over the last 10, 50, 200 events, in basis points |
| rv_200 | realised volatility: sqrt of the sum of squared 1-event log returns over 200 events, in bps |

Short-horizon returns can show momentum (trending flow) or mean reversion (bid-ask bounce). Volatility tells the model how likely any move is, which separates "flat" from "not flat".

Remember: imbalance = state, OFI = flow; microprice tilts the mid toward the thin side; normalise flow by depth; keep features relative (to the mid, spread, or depth) so they mean the same thing at any price.

---

## 5. Labels

**Horizon k.** Predict the direction over the next k = 50 events. Short horizons have more mid changes that are pure noise (bid-ask bounce); long horizons dilute the information in the current book.

**Smoothing.** Instead of comparing m(t+k) with m(t), compare the *average* of the next k mids with m(t):

```
l_t = ( mean(m_{t+1}, ..., m_{t+k}) − m_t ) / m_t
```

This is the labelling used in FI-2010 and DeepLOB. A single price at t+k may just be a bounce; the average is steadier. The downside: the label no longer equals the P&L of a trade exited at t+k, so the backtest computes P&L separately from the actual future mid.

**Three classes.** up if l_t > alpha, down if l_t < −alpha, otherwise flat. Alpha is the 33rd percentile of |l_t| **on the training set only**, which makes the three classes roughly equal there. Choosing alpha from the whole dataset would use test-set information. A small leak, but a classic interview catch.

**Why three classes instead of two?** Most of the time the price does not move enough to matter. Forcing those cases into up or down adds noise. The flat class also gives a natural "do not trade" signal.

Remember: horizon, smoothing, threshold from train only, near-balanced classes.

---

## 6. Time-series splits and leakage

**Why not a random split?** Rows next to each other in time are almost identical (features use overlapping windows, labels overlap). With a random split, the test set contains near-copies of training rows and the score is inflated. Markets also change during the day, so the realistic test is "train on the past, predict the future".

**Chronological 70/15/15.** First 70% of events train, next 15% validate (early stopping, threshold choice), last 15% test. Touch the test set once at the end.

**Purging.** The label at t uses mids up to t + k. The last k training rows therefore have labels built from prices inside the validation period. Drop those k rows at each boundary. A further gap after the boundary is called an **embargo** (López de Prado 2018), which matters when features use long look-back windows.

**Types of leakage to name in an interview.**
1. *Look-ahead in features*: using any data after t (e.g. a centred rolling mean, or the full-day mean for normalising).
2. *Look-ahead in preprocessing*: scaling, thresholds or imputation fitted on all data. Fit them on train only.
3. *Label overlap across splits*: fixed by purging.
4. *Non-stationary features*: like absolute price. Not future information, but the model learns a relation that only held in the training window.
5. *Selection on test*: choosing hyperparameters or trading thresholds by looking at test results. This project picks theta on validation.

**Diagnosing the price-level problem (what the project did).**
* **Gain importance**: price features took 8.5% of total gain on the full model.
* **Adversarial validation**: for each feature, how well does it alone separate train rows from test rows (AUC)? Price features scored about 0.81. Relative features scored 0.50 to 0.55. A high AUC means "the model sees values in test it never saw in train", so trees put all test rows into one edge bin.
* **Permutation importance on test**: shuffling a price feature changed test F1 by about zero, so it carried no out-of-sample information.
* **Ablation**: dropping them lowered train F1 (0.576 → 0.554) and raised test F1 (0.474 → 0.479).

**Walk-forward validation** (what to do with more days): train on days 1..n, test on day n+1, roll forward, average. Gives many test periods, not one.

Remember: purge, fit everything on train, choose thresholds on validation, test once; adversarial AUC is a quick drift detector.

---

## 7. LightGBM

**Gradient boosting.** Build trees one at a time. Each new tree fits the gradient of the loss with respect to the current predictions (for squared error that is the residual). The final prediction is the sum of all trees times a learning rate. Small learning rate + many trees + early stopping is the standard recipe.

**Multiclass.** LightGBM's `multiclass` objective fits one tree per class per round and turns the three scores into probabilities with softmax. The loss is multi-class log-loss (cross-entropy).

**What makes LightGBM fast.**
* *Histogram binning*: each feature is bucketed into at most 255 bins, so split search is over bins, not raw values.
* *Leaf-wise growth*: it splits the leaf with the biggest loss reduction anywhere in the tree, instead of level by level. Faster to a good fit, more prone to overfit, so control it with `num_leaves` and `min_child_samples`.
* *GOSS* and *exclusive feature bundling*: sampling and sparse-feature tricks (not important here).

**Parameters used and why.**
| Parameter | Value | Why |
|---|---|---|
| learning_rate | 0.03 | Small steps, more trees, smoother fit |
| num_leaves | 31 | Limits tree complexity |
| min_child_samples | 500 | Labels are very noisy; a leaf needs many rows to say something real |
| feature_fraction | 0.8 | Each tree sees 80% of features: decorrelates trees |
| bagging_fraction / freq | 0.7 / 1 | Each tree sees 70% of rows |
| lambda_l2 | 5 | Shrinks leaf values |
| early stopping | 100 rounds on validation log-loss | Picks the number of trees (about 100 to 135 here) |

**Why trees and not a linear model?** Microstructure effects interact: imbalance matters more when the spread is one tick; flow matters more when depth is thin. Trees find interactions and thresholds without manual crosses, ignore monotone transforms of inputs, and handle different scales. A logistic regression on OFI and imbalance is still a good baseline to show.

**Why trees cannot extrapolate.** A tree predicts a constant in each leaf. If test prices fall outside the training range, every test row lands in the outermost bin. This is the mechanism behind price-level features failing out of sample.

**Feature importance types.**
* *Split*: how often a feature is used. Biased toward high-cardinality continuous features.
* *Gain*: total loss reduction from its splits. Better, but computed on training data, so it can rank a feature that only helped fit noise.
* *Permutation*: shuffle one column on held-out data and measure the metric drop. Shows out-of-sample value. Correlated features share credit (shuffling one, the model leans on its twin).
* *SHAP*: per-prediction contributions with good theoretical properties. Use `model.predict(X, pred_contrib=True)` in LightGBM.

Remember: boosting fits gradients; leaf-wise growth needs leaf-size limits; early stopping on validation; gain is in-sample, permutation is out-of-sample; trees cannot extrapolate.

---

## 8. Metrics

**Per class:** precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = 2PR / (P + R).

**Averaging over classes.**
* *Macro F1*: average of the three per-class F1s. Every class counts equally.
* *Weighted F1*: weighted by class frequency.
* *Micro F1*: computed from total counts; equals accuracy for single-label multiclass.

**Why macro F1 here.** A model can score well on accuracy by predicting the most common class. Macro F1 punishes ignoring any class, and for trading the up and down classes matter most.

**Random baseline.** Guessing at random with class frequencies p: the expected F1 for class c is about p_c (when train and test frequencies match), so macro F1 ≈ mean(p_c) = 1/3 for three balanced classes. The project measures it with 200 random draws: 0.333 ± 0.002 on test.

**Lift** = model F1 / baseline F1 − 1. Here 0.479 / 0.333 − 1 = 44%.

**Confusion matrix.** Rows = true class, columns = predicted. In this project, flat has the highest recall (59%) against 43% for up and for down. The most common mistake is predicting flat when the price moved, but calling up when it went down (or the reverse) is still frequent. Ask yourself which errors cost money: calling up when it went down costs a lot; calling flat costs only a missed trade.

**Log-loss.** Measures probability quality; it is what early stopping watches. Good probabilities matter because the trading rule uses P(up) − P(down).

**Is 0.48 macro F1 good?** For short-horizon direction on real data, macro F1 in the 0.40s to low 0.50s is typical for tree models on hand-built features. Papers on FI-2010 report higher numbers, but that dataset and its labels are easier. Always compare against a baseline on the *same* labels and split.

---

## 9. Backtesting

**From probabilities to positions.** score = P(up) − P(down). Go long 1 share if score > theta, short if score < −theta, otherwise do nothing. Close after k events.

**Raw P&L (mid to mid).** side × (m_{t+k} − m_t). It measures signal quality as if you could trade at the mid.

**Spread cost.** In reality you buy at the ask and sell at the bid. Relative to the mid you pay half the spread when you enter and half when you exit:

```
cost = s_t / 2 + s_{t+k} / 2
net  = raw − cost
```

Here the average spread is about $0.20, so the round trip costs about $0.20 to $0.25 per share, while the signal earns $0.02 to $0.15 per share. Net P&L is negative at every threshold. That is the most important result to talk about.

**Sharpe ratio.** mean(P&L) / std(P&L). In this project it is per trade. To annualise a *daily* Sharpe you multiply by sqrt(252). Per-trade numbers from overlapping trades on one day should not be annualised: overlapping trades are correlated, so the effective number of independent bets is much smaller than the trade count. The project also reports a non-overlapping version (one position at a time).

**Choosing theta.** Higher theta = fewer, more confident trades with a higher hit rate and bigger average move. Pick it on validation, report it once on test. Here theta = 0.4 was picked: 545 test trades, 79% hit rate, raw per-trade Sharpe 0.55, net −0.56.

**Break-even cost.** The mean raw P&L per trade ($0.15 at theta 0.4) is the most you can pay per round trip and still make money. Ways to get under it:
* trade passively with limit orders (earn the spread instead of paying it, but risk not getting filled and *adverse selection*: you get filled exactly when the price moves against you);
* trade a stock with a tighter spread relative to its moves;
* use a longer horizon so the expected move is bigger than the cost;
* only trade when the spread is tight.

**What this backtest leaves out.** Latency (by the time you act, the book has changed), queue position, market impact of your own orders, exchange fees and rebates, short-selling constraints, position limits.

Remember: raw vs net; cost = half spread in + half spread out; Sharpe per trade is not annual Sharpe; choose thresholds on validation; most short-horizon signals die on costs.

---

## 10. Interview questions and answers

**Q1. Walk me through the project.**
Use the 30-second pitch in section 1. Then offer to go deeper on features, leakage, or the backtest.

**Q2. What is the mid-price and why predict it instead of trade prices?**
The average of best bid and best ask. Trade prices bounce between bid and ask depending on who initiated, which adds noise. The mid is defined at every event and moves only when the best prices change.

**Q3. What is order flow imbalance and why does it predict price changes?**
Net buying pressure at the best quotes: bid queue growth plus ask queue depletion, minus the opposite. A level-1 queue running out is exactly what moves the mid, and OFI measures the flow that empties queues. Cont, Kukanov and Stoikov show that price changes are roughly linear in OFI, with slope inversely related to depth.

**Q4. Difference between queue imbalance and OFI?**
Imbalance is a snapshot of the current queues. OFI is the sum of changes over a window, including price-level shifts. They are correlated but not the same. OFI captures aggressive flow and cancellations; imbalance captures the standing pressure.

**Q5. What is the microprice?**
(ask × bid size + bid × ask size) / (bid size + ask size). It moves the fair price toward the side with the smaller queue, because that side is more likely to be depleted. Stoikov's microprice refines this with a model of how imbalance and spread evolve.

**Q6. Why did you remove absolute-price features?**
Price level is non-stationary. The tree learned rules like "between $586 and $587 the price rose" that held in the training hours only. Evidence: 8.5% of gain, AUC about 0.81 for telling train from test rows, about zero permutation importance on test, and removing them raised test F1 slightly and narrowed the train–test gap. Relative features (normalised by mid, spread or depth) keep the same meaning at any price.

**Q7. How would you detect that kind of problem in general?**
Adversarial validation (train a classifier to tell train from test, look at which features it uses), compare gain importance with held-out permutation importance, plot features over time, ablation studies, and walk-forward evaluation.

**Q8. Why a time-based split? What is purging?**
Neighbouring rows share information, and the real use case is predicting the future. Random splits leak near-duplicates into test. Purging drops the last k training rows before each boundary because their labels look k events ahead into the next split.

**Q9. How did you choose the label threshold, and why does it matter where you compute it?**
The 33rd percentile of absolute future return on the training set, giving balanced classes. Computing it on the full data uses the test period's volatility, which is a leak.

**Q10. Why macro F1 instead of accuracy?**
Accuracy can be high by favouring one class. Macro F1 weights up, down and flat equally, and the directional classes are the ones that matter for trading.

**Q11. What is the random baseline's macro F1 and why?**
About 1/3. With random guessing at class frequency p_c, expected precision and recall for class c are both about p_c, so F1 ≈ p_c, and the macro average of balanced classes is 1/3.

**Q12. Why LightGBM over a neural network like DeepLOB?**
Tabular, hand-built features with interactions, one day of data, need for fast iteration and interpretability. Gradient-boosted trees are a strong baseline there. DeepLOB-style CNN/LSTM models learn features from raw book snapshots and need much more data and tuning. A fair comparison would be on the same labels and split.

**Q13. How do you prevent LightGBM from overfitting noisy labels?**
Large `min_child_samples`, limited `num_leaves`, small learning rate with early stopping on a time-ordered validation set, row and feature subsampling, L2 regularisation. Then check the train vs validation gap.

**Q14. Gain importance vs permutation importance vs SHAP?**
Gain is in-sample loss reduction, so it can credit features that fit noise. Permutation importance is the metric drop on held-out data when a column is shuffled; it is out-of-sample but splits credit between correlated features. SHAP gives additive per-prediction attributions; mean |SHAP| is a good global ranking.

**Q15. Your top feature is spread. Is that a problem?**
No, but understand what it does. Spread mostly predicts *whether* the mid moves (wide spread, bigger jumps when it closes), not *which way*. Direction comes from cancel, trade and submit imbalance. The confusion matrix agrees: flat has 59% recall, up and down 43% each.

**Q16. How did you compute the backtest P&L and costs?**
Long or short one share when P(up) − P(down) passes theta, exit k events later. Raw = side × mid change. Cost = half spread at entry + half spread at exit. Theta chosen on validation.

**Q17. Your strategy loses money after costs. Is the project a failure?**
No. It shows the signal is real (raw Sharpe 0.55 per trade, 79% hit rate at the chosen threshold) and quantifies exactly how much it is worth ($0.15 per share per trade) against what it costs to take liquidity ($0.25). That tells you what has to change: passive execution, tighter-spread names, longer horizons, or conditioning on tight spreads.

**Q18. What is adverse selection for a passive order?**
Your limit order is more likely to fill when the market is moving through your price, that is, when the price is about to move against you. Passive fills save the spread but are biased toward bad outcomes. A passive strategy must model fill probability and post-fill drift.

**Q19. Why not annualise the Sharpe?**
The trades overlap and come from one day, so they are strongly correlated. Annualising per-trade Sharpe by sqrt(trades per year) massively overstates it. You need daily P&L over many days, then multiply the daily Sharpe by sqrt(252).

**Q20. Event time vs clock time sampling?**
Event time adapts to activity and gives returns closer to normally distributed. Clock time is easier to align across assets and to execute on. A 50-event horizon is a few seconds when busy and longer when quiet.

**Q21. What other leakage could exist in a project like this?**
Features using future rows (centred windows, full-day normalisation), labels that overlap the split, thresholds fit on all data, selecting the model or theta by looking at test, and in real data, timestamps that are not the times you could have acted (latency).

**Q22. How would you improve the model?**
More days with walk-forward validation; multiple stocks; per-stock normalisation; horizon in seconds; time-of-day features; SHAP-based feature selection; probability calibration; a cost-aware objective (predict moves larger than the spread); DeepLOB-style raw-book models as comparison.

**Q23. How would you make it tradable?**
Predict moves net of the spread (label = move > cost); trade only when the spread is tight; use passive orders with a fill-probability model; add inventory and risk limits; simulate latency and queue position; measure on many days out of sample.

**Q24. What does a hidden execution (type 5) tell you?**
A trade happened against an order not shown in the book. It is real trading pressure with no visible book change, so it goes into trade-flow features but not into OFI.

**Q25. Why is the trade direction in LOBSTER inverted?**
The direction field always describes the resting limit order. In an execution, the resting order was hit by the aggressor, so the aggressor's sign is the opposite.

**Q26. How would results change on a large-tick stock?**
The spread is almost always one tick, so the mid moves rarely; the "flat" class dominates at short horizons, so you would need longer horizons or a different threshold. Queue imbalance becomes much more predictive. Costs relative to moves are higher.

**Q27. You used simulated data. How do you know the pipeline works on real data?**
The simulator writes the exact LOBSTER layout, and the loader and features were checked on a real AAPL LOBSTER excerpt. The simulator matches the real sample's message mix and spread level. The absolute model numbers are not evidence about real markets, and I would rerun on real LOBSTER data before quoting them.

**Q28. What would you check if test F1 was much higher than validation F1?**
Leakage first: overlapping labels, features using future data, a bug in the split. Then whether the test period is simply easier (lower noise, different volatility). Compare class balance and feature distributions across periods.

**Q29. Bias-variance in this setting?**
Labels are very noisy, so variance dominates: complex models fit noise. That is why leaves are large, the learning rate is small, and early stopping kicks in after about 100 trees.

**Q30. Explain gradient boosting in two sentences.**
Start from a constant prediction, then repeatedly fit a small tree to the gradient of the loss with respect to the current predictions and add a shrunken copy of it. Each tree corrects the remaining errors of the ensemble.

---

## 11. Further reading

* Cont, Kukanov, Stoikov (2014). *The Price Impact of Order Book Events.* OFI.
* Stoikov (2018). *The micro-price: a high-frequency estimator of future prices.*
* Gould, Bonart (2016). *Queue imbalance as a one-tick-ahead price predictor in a limit order book.*
* Ntakaris et al. (2018). *Benchmark dataset for mid-price forecasting of limit order book data* (FI-2010).
* Zhang, Zohren, Roberts (2019). *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.*
* Kolm, Turiel, Westray (2023). *Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book.*
* López de Prado (2018). *Advances in Financial Machine Learning*, chapters 3 (labelling) and 7 (purged cross-validation).
* Bouchaud, Bonart, Donier, Gould (2018). *Trades, Quotes and Prices.* The textbook on market microstructure.
* Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.*
* LOBSTER documentation: lobsterdata.com/info/DataStructure.php
