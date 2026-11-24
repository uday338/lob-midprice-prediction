# Data

## Simulated (default)

```bash
python src/simulate_lobster.py --n-events 400000 --seed 7
```
writes `data/simulated/SIMX_2012-06-21_34200000_57600000_{message,orderbook}_10.csv` (about 107 MB, not committed).
Seed 7 reproduces the numbers in the README.

## Real LOBSTER data

1. Go to https://lobsterdata.com/info/DataSamples.php and download a free sample, e.g.
   `LOBSTER_SampleFile_AAPL_2012-06-21_10.zip` (AAPL, 21 June 2012, 10 levels).
2. Unzip it into its own folder, e.g. `data/lobster/`. You need one `*_message_10.csv` and one
   `*_orderbook_10.csv` in that folder.
3. Run `python src/run_pipeline.py --data-dir data/lobster --out-dir results_real`.

File layout (no header row):
* message: `time` (seconds after midnight), `type` (1 submit, 2 partial cancel, 3 delete,
  4 visible execution, 5 hidden execution, 6 cross trade, 7 halt), `order_id`, `size`,
  `price` ($ x 10000), `direction` (1 buy order, -1 sell order).
* orderbook: `ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...` up to level 10, giving the
  book right after the message on the same line. Empty levels use dummy prices
  9999999999 / -9999999999.

The loader drops types 6 and 7 and turns dummy prices into empty levels.
