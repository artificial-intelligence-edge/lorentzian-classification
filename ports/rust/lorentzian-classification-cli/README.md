# lorentzian-classification-cli

[![crates.io](https://img.shields.io/crates/v/lorentzian-classification-cli?style=flat-square&labelColor=4A4A4A)](https://crates.io/crates/lorentzian-classification-cli)
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square&labelColor=4A4A4A)](https://github.com/artificial-intelligence-edge/lorentzian-classification/blob/main/LICENSE.md)

Command-line interface for the Rust port of the
[Lorentzian Classification](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/)
indicator. A thin front end over
[`lorentzian-classification-core`](https://crates.io/crates/lorentzian-classification-core)
with no third-party argument parsing, mirroring the essential verbs of the
repository's stdlib-only Python CLI.

## Installation

```bash
cargo install lorentzian-classification-cli
```

The installed binary is named `lorentzian-classification`.

## Usage

```bash
# Compute the full 40-column result series from a TradingView export.
lorentzian-classification run input.csv output.csv

# Extend the neighbor scan across the chart's full history.
lorentzian-classification run input.csv output.csv --include-full-history

# Recompute from a Pine export and compare against its own columns
# (features/kernel within 1e-6; predictions and signals exact).
lorentzian-classification parity pine_export.csv
```

`run` reads the `time,open,high,low,close` TradingView schema (quoted cells
supported), derives the ADX price scale from the data's decimal precision,
and writes the same 40-column schema as the Python port. `parity` exits
non-zero on any mismatch, so it can gate pipelines.

See the [repository](https://github.com/artificial-intelligence-edge/lorentzian-classification)
for the PineScript reference, the other ports, and the validation
methodology.

Licensed under the [MIT License](https://github.com/artificial-intelligence-edge/lorentzian-classification/blob/main/LICENSE.md).
