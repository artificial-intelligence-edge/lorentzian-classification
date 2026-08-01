# Rust Port

[![crates.io](https://img.shields.io/crates/v/lorentzian-classification-core?style=flat-square&labelColor=4A4A4A&label=crates.io)](https://crates.io/crates/lorentzian-classification-core)
[![docs.rs](https://img.shields.io/docsrs/lorentzian-classification-core?style=flat-square&labelColor=4A4A4A)](https://docs.rs/lorentzian-classification-core)

A bit-faithful Rust port of the Lorentzian Classification indicator, structured
as a Cargo workspace:

| Crate | Purpose |
| --- | --- |
| [`lorentzian-classification-core`](lorentzian-classification-core/) | The algorithm as a library. Zero dependencies with default features off; the default `csv` feature adds TradingView CSV input and Pine-export parity checking. |
| [`lorentzian-classification`](lorentzian-classification/) | Alias crate re-exporting the core under the canonical name. |
| [`lorentzian-classification-cli`](lorentzian-classification-cli/) | A thin command-line front end (`run`, `parity`). |

The port is a statement-for-statement translation of the PineScript v6 reference
and the parity-tested Python port (`ports/python`). Series are forward-indexed
(index `0` = oldest bar) and Pine `na` is represented as `f64::NAN`.

## Using the published crates

```bash
cargo add lorentzian-classification-core   # library (or: lorentzian-classification)
cargo install lorentzian-classification-cli   # CLI binary: lorentzian-classification
```

```toml
[dependencies]
lorentzian-classification-core = "0.1"
# or, for the zero-dependency numeric core:
lorentzian-classification-core = { version = "0.1", default-features = false }
```

See the [core crate README](lorentzian-classification-core/README.md) for the
consumer-facing quick start and the Pine-input-to-`Settings` mapping.

## Status

**Implemented, parity-verified, and published.** The port reproduces the
TradingView/Pine gold exports in `tests/parity/baselines/` exactly, under the
same contract used for the Python port (`1e-6` for features/kernel, exact for
prediction/direction/buy/sell/stops). See [Validation](#validation).

## Design

- **Small dependency surface.** The numeric core has zero runtime
  dependencies; the default `csv` feature adds the established `csv` crate so
  quoted fields behave like Python's standard `csv` parser instead of relying
  on ad hoc string splitting.
- **`#![forbid(unsafe_code)]`**, `#![deny(missing_docs)]`, and
  `#![warn(clippy::pedantic)]` at the crate root; the workspace passes
  `clippy -D warnings` and `rustfmt --check`.
- **Bit-exact floating point.** Banker's rounding (`f64::round_ties_even`) is
  used for the ADX price-scale quantization to match Python's `round`, and
  `powf`/`exp` match libm, so the two ports agree to the last bit.
- **Typed configuration.** [`Settings`], [`Source`], and [`FeatureSpec`] encode
  the Pine inputs; `Settings::default()` equals the Pine defaults.
- **Self-contained package.** One gold baseline is vendored into the core
  crate (`tests/data/`), so the crates.io package passes its parity suite
  outside the monorepo; a drift-guard test keeps the vendored copy
  byte-identical to the monorepo baseline.

Modules: `types`, `indicators` (RSI/WT/CCI/ADX, SMA/EMA/RMA), `kernels`
(Rational Quadratic, Gaussian), `filters` (regime/KLMF), `ann` (the Lorentzian
approximate-nearest-neighbor scan), `engine` (`calculate`), `display` (Pine
colors), `csv_io`, and `parity` (behind the `csv` feature).

## Library usage

```rust
use lorentzian_classification_core::{calculate, read_tradingview_csv, Settings};

let (bars, price_scale) = read_tradingview_csv("data.csv".as_ref())?;
let rows = calculate(&bars, &Settings::default(), price_scale);
println!("last prediction: {}", rows.last().unwrap().prediction);
# Ok::<(), lorentzian_classification_core::CsvError>(())
```

## CLI usage

```bash
# Compute the full 40-column result series.
cargo run --release -p lorentzian-classification-cli -- \
  run "input.csv" "output.csv" --include-full-history

# Recompute from a Pine export and compare against its own columns.
cargo run --release -p lorentzian-classification-cli -- \
  parity "tests/parity/baselines/pine_oanda_eurusd_1d_full_history.csv" \
  --include-full-history --tolerance 1e-6
```

## Validation

```bash
cd ports/rust

cargo test --release --workspace          # parity + unit tests
cargo test -p lorentzian-classification-core --no-default-features   # zero-dependency core
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS='-D warnings -D missing_docs' cargo doc --workspace --no-deps
cargo bench                               # criterion benchmarks for `calculate`
```

The parity integration test (`lorentzian-classification-core/tests/parity.rs`)
runs the Rust port against the committed gold baselines. Because the Python port
is already proven equal to those same Pine exports, passing here establishes
**Rust == Pine == Python** transitively.

For an end-to-end cross-port check across the full output schema (including
columns the Pine exports omit: backtest stream, alerts, colors, trade stats),
run the repository harness:

```bash
tests/parity/cross_port_parity.sh
```

It runs both the Rust and Python CLIs on each baseline and diffs the outputs with
an implementation-independent comparator.

## Releasing

Releases are tag-driven and published with crates.io Trusted Publishing (see
`.github/workflows/release-rust.yml`):

1. Bump `workspace.package.version` in `Cargo.toml` and the
   `lorentzian-classification-core` entry under `workspace.dependencies`.
2. Update each crate's `CHANGELOG.md`.
3. Land the change on `main`, then tag it `rust-vX.Y.Z` and push the tag.

The workflow verifies the workspace (fmt, clippy, tests with and without
default features, the docs gate, the MSRV check against the committed
lockfile, and an extracted-package test outside the workspace), then publishes
any not-yet-published crate versions in dependency order. Versioned releases
are immutable on crates.io; a partially failed run can be re-run and skips
what already published.
