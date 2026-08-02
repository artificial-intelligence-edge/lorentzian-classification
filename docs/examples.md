# Examples

Hands-on, copy-pasteable workflows that exercise every implemented port in this
repository.

Every command below runs **from the repository root** and uses the
TradingView/Pine gold exports already committed under
`tests/parity/baselines/`. No external data, accounts, or API keys required.

The MQL5 / MetaTrader 5 port lives under `ports/mql5/`; it runs inside
MetaTrader 5 rather than from the repository, so it has no repo-runnable
examples until it joins the parity workflow.

## 1. Run the indicator on real market data (Python)

Compute the full 40-column result series (features, kernel estimate,
prediction, direction, buy/sell/exit signals, alerts, backtest stream, and
trade stats) for ~3 years of Coinbase BTC/USD daily bars:

```bash
PYTHONPATH=ports/python python3 -m lorentzian_classification run \
  tests/parity/baselines/pine_coinbase_btcusd_1d_limited_history.csv \
  --output /tmp/btcusd_daily_signals.csv
```

Then inspect the most recent signals:

```bash
column -s, -t < /tmp/btcusd_daily_signals.csv | tail -5
```

Tune the model exactly like the TradingView inputs: every Pine `input.*` has a
matching CLI flag:

```bash
PYTHONPATH=ports/python python3 -m lorentzian_classification run \
  tests/parity/baselines/pine_coinbase_btcusd_1d_limited_history.csv \
  --neighbors 5 \
  --f1 RSI:14:1 --f2 WT:10:11 --f3 CCI:20:1 \
  --output /tmp/btcusd_custom.csv
```

The full list of settings and what they do is documented in the
[settings reference](https://ai-edge.io/docs/indicators/lorentzian-classification/general-settings).

## 2. Use it as a Python library

```python
from lorentzian_classification import LorentzianClassification

model = LorentzianClassification(
    "tests/parity/baselines/pine_coinbase_btcusd_1d_limited_history.csv"
)
latest = model.latest
print(latest.bar.time, latest.prediction, latest.direction, latest.buy, latest.sell)

model.to_csv("/tmp/signals.csv")        # full 40-column artifact
df = model.to_dataframe()               # pip install "lorentzian-classification[dataframe]"
```

The wrapper also accepts `Bar` iterables, mapping records, or any
DataFrame-like object with `to_dict("records")`. See the Library API section of
[`ports/python/README.md`](../ports/python/README.md) for the full surface.

## 3. Same answer, compiled speed (Rust)

Install the published 0.1.0 CLI and run it directly:

```bash
cargo install lorentzian-classification-cli --version 0.1.0 --locked
lorentzian-classification run \
  tests/parity/baselines/pine_oanda_eurusd_1d_full_history.csv \
  /tmp/eurusd_rust.csv --include-full-history
```

Library consumers can start with `cargo add lorentzian-classification-core`;
the [core API documentation](https://docs.rs/lorentzian-classification-core)
includes examples for direct `Bar` input and TradingView CSVs. To run the CLI
from a repository checkout instead:

```bash
cargo run --release \
  --manifest-path ports/rust/Cargo.toml -p lorentzian-classification-cli -- \
  run tests/parity/baselines/pine_oanda_eurusd_1d_full_history.csv \
  /tmp/eurusd_rust.csv --include-full-history
```

The Rust port is bit-exact with the Python port: not "close", identical to
the last bit of every float (banker's rounding and libm-matched `powf`/`exp`).
The crates.io release was also tested from fresh downstream projects rather
than relying only on the monorepo workspace.

## 4. Same answer, formally specified (Lean 4)

The Lean port is an executable formal specification: the algorithm plus proved
structural invariants. Its CLI is drop-in compatible with the other two:

```bash
cd ports/lean && lake build
.lake/build/bin/lorentzian-classification \
  run ../../tests/parity/baselines/pine_oanda_eurusd_1d_full_history.csv \
  /tmp/eurusd_lean.csv
```

Its output is **byte-identical** to the Rust port on all four gold baselines.

## 5. Don't trust us: prove parity yourself

Recompute from a Pine export's own OHLC columns and diff against the values
TradingView itself produced (features/kernel within `1e-6`,
prediction/direction/buy/sell exact):

```bash
PYTHONPATH=ports/python python3 -m lorentzian_classification parity \
  tests/parity/baselines/pine_oanda_eurusd_1d_full_history.csv
```

Run the whole committed fixture suite, including manifest-pinned source hashes:

```bash
PYTHONPATH=ports/python python3 -m lorentzian_classification validate-fixtures
```

Prove all three implemented ports agree with each other on the full 40-column
schema (builds Rust and Lean, runs every baseline through Python + Rust + Lean,
and diffs with the stdlib-only comparator so the check is not circular):

```bash
tests/parity/cross_port_parity.sh
```

Diff any two output CSVs yourself:

```bash
python3 tests/parity/compare_csv.py /tmp/eurusd_rust.csv /tmp/eurusd_lean.csv --tolerance 1e-9
```

## 6. Bring your own TradingView data

1. Add the indicator to your chart:
   [Machine Learning: Lorentzian Classification](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/).
2. Generate the plot snippet that exposes the parity columns in TradingView's
   CSV export, and paste it into the Pine editor:

   ```bash
   PYTHONPATH=ports/python python3 -m lorentzian_classification pine-export-helper
   ```

3. Export chart data as CSV (`⋮` → *Export chart data…*), then run any port on
   it, or run `parity` against it to verify your local results match your
   chart bar-for-bar.

## 7. Go beyond the defaults

- **Settings reference**: what every input does and how to transfer settings
  between platforms:
  [ai-edge.io/docs](https://ai-edge.io/docs/indicators/lorentzian-classification/general-settings)
- **Optimizer**: hosted parameter-optimization studies; take a study's best
  trial and replay it locally with the matching CLI flags
  (`--neighbors`, `--f1 RSI:14:1`, kernel parameters, filter toggles):
  [optimizer.ai-edge.io/studies](https://optimizer.ai-edge.io/studies)
- **Coverage matrix**: which non-default settings are already parity-proved
  and which Pine exports are still wanted:
  `tests/parity/python_port_coverage.md`
