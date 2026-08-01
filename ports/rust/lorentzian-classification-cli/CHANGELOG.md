# Changelog

All notable changes to `lorentzian-classification-cli` are documented here.

## 0.1.0

Initial crates.io release.

- `run <input.csv> <output.csv> [--include-full-history]`: compute the full
  40-column result series from a TradingView export.
- `parity <pine_export.csv> [--include-full-history]`: recompute from a Pine
  export and compare against its own columns; non-zero exit on mismatch.
- CSV formula-injection neutralization on output cells, matching the Python
  CLI's writer behavior.
