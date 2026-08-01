# Changelog

All notable changes to `lorentzian-classification-core` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the crate adheres to [Semantic Versioning](https://semver.org/).

## 0.1.0

Initial crates.io release.

- Bit-faithful port of the Lorentzian Classification PineScript v6 reference:
  feature engineering (RSI, WT, CCI, ADX), rational quadratic and Gaussian
  kernel regression, volatility/regime/ADX/EMA/SMA filters, the
  approximate-nearest-neighbors Lorentzian scan, entries/exits, alerts, and
  trade statistics.
- Parity-tested against committed TradingView gold exports (features and
  kernel within `1e-6`; predictions, directions, and signals exact) and
  bit-for-bit equal to the repository's Python port.
- `csv` cargo feature (default): TradingView CSV input and the Pine-export
  parity comparator. With `default-features = false` the crate has zero
  dependencies.
- One gold baseline vendored into the package so the extracted crate passes
  its parity suite outside the monorepo.
- Documented public API (`#![deny(missing_docs)]`), `#![forbid(unsafe_code)]`.
