# lorentzian-classification

[![crates.io](https://img.shields.io/crates/v/lorentzian-classification?style=flat-square&labelColor=4A4A4A)](https://crates.io/crates/lorentzian-classification)
[![docs.rs](https://img.shields.io/docsrs/lorentzian-classification?style=flat-square&labelColor=4A4A4A)](https://docs.rs/lorentzian-classification)
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square&labelColor=4A4A4A)](https://github.com/artificial-intelligence-edge/lorentzian-classification/blob/main/LICENSE.md)

The [Lorentzian Classification](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/)
indicator by [Justin Dehorty](https://www.tradingview.com/u/jdehorty/) for
Rust, under its canonical name.

This crate is a thin alias: it re-exports the entire public surface of
[`lorentzian-classification-core`](https://crates.io/crates/lorentzian-classification-core),
where the implementation, documentation, and parity tests live. Depend on
whichever name reads better in your `Cargo.toml`; both resolve to the same
code.

```toml
[dependencies]
lorentzian-classification = "0.1"
```

```rust
use lorentzian_classification::{calculate, Bar, Settings};

let bars: Vec<Bar> = fetch_bars();
let rows = calculate(&bars, &Settings::default(), 0.0);
println!("last prediction: {}", rows.last().unwrap().prediction);
```

The `csv` cargo feature (default) forwards to the core crate's `csv` feature:
TradingView CSV input and Pine-export parity checking. With
`default-features = false` the dependency tree is the zero-dependency numeric
core.

See the [core crate](https://crates.io/crates/lorentzian-classification-core)
for the full README, and the
[repository](https://github.com/artificial-intelligence-edge/lorentzian-classification)
for the PineScript reference, the Python and Lean 4 ports, and the validation
methodology.

Licensed under the [MIT License](https://github.com/artificial-intelligence-edge/lorentzian-classification/blob/main/LICENSE.md).
