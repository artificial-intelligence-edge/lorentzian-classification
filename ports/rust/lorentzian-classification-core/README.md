# lorentzian-classification-core

[![crates.io](https://img.shields.io/crates/v/lorentzian-classification-core?style=flat-square&labelColor=4A4A4A)](https://crates.io/crates/lorentzian-classification-core)
[![docs.rs](https://img.shields.io/docsrs/lorentzian-classification-core?style=flat-square&labelColor=4A4A4A)](https://docs.rs/lorentzian-classification-core)
[![CI](https://img.shields.io/github/actions/workflow/status/artificial-intelligence-edge/lorentzian-classification/ci.yml?style=flat-square&labelColor=4A4A4A&label=CI)](https://github.com/artificial-intelligence-edge/lorentzian-classification/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square&labelColor=4A4A4A)](https://github.com/artificial-intelligence-edge/lorentzian-classification/blob/main/LICENSE.md)

A bit-faithful Rust port of the
[Lorentzian Classification](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/)
indicator by [Justin Dehorty](https://www.tradingview.com/u/jdehorty/): an
approximate-nearest-neighbors classifier over Lorentzian distance, with the
full feature engineering, kernel regression, filter, signal, and trade-stats
pipeline of the PineScript v6 reference.

The port is a statement-for-statement translation of the reference and is
parity-tested against committed TradingView gold exports: features and kernel
estimates agree within `1e-6`, and predictions, directions, and entry/exit
signals match exactly. It is also bit-for-bit equal to the
[Python port](https://github.com/artificial-intelligence-edge/lorentzian-classification/tree/main/ports/python)
of the same repository. One gold baseline is vendored into the crate, so
`cargo test` proves parity even from the extracted package.

## Installation

```bash
cargo add lorentzian-classification-core
```

Or in `Cargo.toml`:

```toml
[dependencies]
lorentzian-classification-core = "0.1"
```

## Quick start with your own data

The core API is two types and one function: build a `Vec<Bar>`, pick
`Settings`, call `calculate`.

```rust
use lorentzian_classification_core::{calculate, Bar, Settings};

let bars: Vec<Bar> = my_feed
    .iter()
    .map(|candle| Bar {
        time: candle.timestamp.to_string(),
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
    })
    .collect();

// price_scale quantizes ADX at the instrument's decimal precision
// (10^decimals); pass 0.0 to disable quantization.
let rows = calculate(&bars, &Settings::default(), 100_000.0);
let last = rows.last().unwrap();
println!("prediction {} direction {}", last.prediction, last.direction);
```

Series are forward-indexed (index `0` is the oldest bar) and missing values
are `f64::NAN`, exactly like Pine `na`.

## Quick start with a TradingView CSV

With the default `csv` feature, exported TradingView data loads directly and
the price scale is derived from the data:

```rust
use lorentzian_classification_core::{calculate, read_tradingview_csv, Settings};

let (bars, price_scale) = read_tradingview_csv("export.csv".as_ref())?;
let rows = calculate(&bars, &Settings::default(), price_scale);
# Ok::<(), lorentzian_classification_core::CsvError>(())
```

## Configuring settings

`Settings::default()` reproduces the TradingView defaults. Every field maps to
a named Pine input (documented field by field on
[`Settings`](https://docs.rs/lorentzian-classification-core/latest/lorentzian_classification_core/types/struct.Settings.html)),
so a chart configuration can be transcribed directly. Start from the default
and override what the chart changes:

```rust
use lorentzian_classification_core::{FeatureKind, FeatureSpec, Settings, Source};

let settings = Settings {
    source: Source::Close,                            // "Source"
    neighbors_count: 12,                              // "Neighbors Count"
    max_bars_back: 1000,                              // "Max Bars Back"
    use_adx_filter: true,                             // "Use ADX Filter"
    f1: FeatureSpec::new(FeatureKind::Rsi, 14, 1),    // "Feature 1", A, B
    f2: FeatureSpec::new(FeatureKind::Wt, 10, 11),    // "Feature 2", A, B
    ..Settings::default()
};
```

Feature specs also parse from the `KIND:paramA:paramB` string form used by
the ports' command-line interfaces: `FeatureSpec::parse("CCI:20:1")`.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `csv` | yes | `read_tradingview_csv` and the `parity` module for checking recomputations against Pine export CSVs, via the `csv` crate. |

With `default-features = false` the crate has **zero dependencies**: just the
numeric core, for consumers that feed `Bar` values from their own pipeline.

## Validation

The parity contract, gold baselines, and cross-port equivalence (PineScript,
Python, Rust, Lean 4) are documented in the
[repository validation notes](https://github.com/artificial-intelligence-edge/lorentzian-classification/blob/main/docs/validation.md).
The `parity` module exposes the same comparator used by the test suite, so
downstream integrations can verify their own exports.

## Related crates

| Crate | Purpose |
| --- | --- |
| [`lorentzian-classification`](https://crates.io/crates/lorentzian-classification) | Alias crate re-exporting this one under the canonical name. |
| [`lorentzian-classification-cli`](https://crates.io/crates/lorentzian-classification-cli) | Command-line interface (`run`, `parity`). |

## Minimum supported Rust version

Rust `1.77`. MSRV bumps are considered semver-minor and verified in CI with
the committed lockfile.

## Safety and license

`#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, clippy pedantic clean.
Licensed under the [MIT License](https://github.com/artificial-intelligence-edge/lorentzian-classification/blob/main/LICENSE.md).

This is analysis software, not investment advice; see the repository README
for the project's scope and disclaimers.
