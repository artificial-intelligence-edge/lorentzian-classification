<div align="center">

<h1>Lorentzian Classification</h1>

<p>
  <strong>The <a href="https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/">Lorentzian Classification</a> indicator by <a href="https://www.tradingview.com/u/jdehorty/">Justin Dehorty</a>,<br>ported to Python, Rust, Lean 4, and MQL5.</strong>
</p>

<!-- tradingview-badges:start -->
<p>
  <a href="https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/"><img alt="TradingView Editors' Picks" src="https://img.shields.io/badge/TradingView-Editors%27%20Picks-2962FF?style=for-the-badge&amp;logo=tradingview&amp;logoColor=white&amp;labelColor=4A4A4A"></a>
</p>
<p>
  <a href="https://www.tradingview.com/chart/BTCUSD/LYCOEW6Z-TradingView-Community-Awards-2023/"><img alt="Community Awards: Most Valuable PineScript 2023" src="https://img.shields.io/badge/Community%20Awards-Most%20Valuable%20PineScript%20%282023%29-2962FF?style=flat-square&amp;logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xOSA1aC0yVjNIN3YySDVjLTEuMSAwLTIgLjktMiAydjFjMCAyLjU1IDEuOTIgNC42MyA0LjM5IDQuOTQuNjMgMS41IDEuOTggMi42MyAzLjYxIDIuOTZWMTlIN3YyaDEwdi0yaC00di0zLjFjMS42My0uMzMgMi45OC0xLjQ2IDMuNjEtMi45NkMxOS4wOCAxMi42MyAyMSAxMC41NSAyMSA4VjdjMC0xLjEtLjktMi0yLTJ6TTUgOFY3aDJ2My44MkM1Ljg0IDEwLjQgNSA5LjMgNSA4em0xNCAwYzAgMS4zLS44NCAyLjQtMiAyLjgyVjdoMnYxeiIvPjwvc3ZnPg%3D%3D&amp;logoColor=white&amp;labelColor=4A4A4A"></a>
</p>
<p>
  <a href="https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/"><img alt="1,184,716 TradingView views" src="https://img.shields.io/badge/Views-1.18M-2962FF?style=flat-square&amp;logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiA0LjVDNyA0LjUgMi43MyA3LjYxIDEgMTJjMS43MyA0LjM5IDYgNy41IDExIDcuNXM5LjI3LTMuMTEgMTEtNy41Yy0xLjczLTQuMzktNi03LjUtMTEtNy41ek0xMiAxN2MtMi43NiAwLTUtMi4yNC01LTVzMi4yNC01IDUtNSA1IDIuMjQgNSA1LTIuMjQgNS01IDV6bTAtOGMtMS42NiAwLTMgMS4zNC0zIDNzMS4zNCAzIDMgMyAzLTEuMzQgMy0zLTEuMzQtMy0zLTN6Ii8%2BPC9zdmc%2B&amp;logoColor=white&amp;labelColor=4A4A4A"></a>
  <a href="https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/"><img alt="35,076 TradingView boosts" src="https://img.shields.io/badge/Boosts-35.1K-2962FF?style=flat-square&amp;logo=rocket&amp;logoColor=white&amp;labelColor=4A4A4A"></a>
  <a href="https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/"><img alt="814 TradingView comments" src="https://img.shields.io/badge/Comments-814-2962FF?style=flat-square&amp;logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0yMCAySDRjLTEuMSAwLTIgLjktMiAydjE4bDQtNGgxNGMxLjEgMCAyLS45IDItMlY0YzAtMS4xLS45LTItMi0yeiIvPjwvc3ZnPg%3D%3D&amp;logoColor=white&amp;labelColor=4A4A4A"></a>
</p>
<!-- tradingview-badges:end -->

<p>
  <a href="ports/pinescript/"><img alt="PineScript v6 port" src="https://img.shields.io/badge/PineScript%20v6-%E2%9C%93-2E7D32?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAyTDE1IDdMMTMuNSA3TDE3IDEyTDE1IDEyTDE5IDE4TDEzLjUgMThMMTMuNSAyMkwxMC41IDIyTDEwLjUgMThMNSAxOEw5IDEyTDcgMTJMMTAuNSA3TDkgN1oiLz48L3N2Zz4%3D&logoColor=white&labelColor=4A4A4A"></a>
  <a href="ports/python/"><img alt="Python port" src="https://img.shields.io/badge/Python-%E2%9C%93-2E7D32?style=flat-square&logo=python&logoColor=white&labelColor=4A4A4A"></a>
  <a href="ports/rust/"><img alt="Rust port" src="https://img.shields.io/badge/Rust-%E2%9C%93-2E7D32?style=flat-square&logo=rust&logoColor=white&labelColor=4A4A4A"></a>
  <a href="ports/lean/"><img alt="Lean 4 port" src="https://img.shields.io/badge/Lean%204-%E2%9C%93-2E7D32?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTYgMjU2IiBmaWxsPSJ3aGl0ZSI%2BPHBhdGggZD0iTTU1IDQ0aDEwdjY0aDM4djEwSDU1Wk0xNTEgNDRoNDh2MTBoLTQ4Wk0xNjEgNzRoMzh2MTBoLTM4Wk0xNTEgMTEyaDQ4djEwaC00OFpNMTg5IDQ0aDEwdjc4aC0xMFpNNTUgMTM5aDEwbDIyIDcyaC0xMFpNMTA5IDEzOWgxMGwtMjIgNzJoLTEwWk02MyAxNjZoNDJ2OUg2M1pNMTQzIDEzOWgxMHY3MmgtMTBaTTE5MCAxMzloMTB2NzJoLTEwWk0xNTIgMTM5aDhsMzggNzJoLTlaIi8%2BPC9zdmc%2B&logoColor=white&labelColor=4A4A4A"></a>
  <a href="ports/mql5/"><img alt="MQL5 port" src="docs/assets/mql5-badge.svg"></a>
</p>

<p>
  <em>Don't see the language you're looking for? Request a new port <a href="https://github.com/artificial-intelligence-edge/lorentzian-classification/issues/new?template=port-request.yml">here</a>.</em>
  <br><br>
  <a href="https://ai-edge.io"><img alt="AI Edge" src="docs/assets/ai-edge-logo.png" width="40"></a><br>
  <sup>Maintained by <a href="https://ai-edge.io">AI Edge</a>.</sup>
</p>

</div>

[![Machine Learning: Lorentzian Classification on TradingView](docs/assets/lorentzian-classification-chart.png)](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/)

<p align="center"><em><a href="https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/">View live on TradingView →</a></em></p>

## Overview

Lorentzian Classification is an open-source TradingView indicator: a
**nearest-neighbor-based classifier** (it labels the current bar by looking at
the historical bars it most resembles) that uses **Lorentzian distance** as the
measure of resemblance to model price movement. It is a classifier, not deep
learning and not an autonomous trading agent.

- **Want to use it on a chart?** Add the original indicator on TradingView:
  [Machine Learning: Lorentzian Classification](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/).
- **Want to use it on MetaTrader 5?** Start with the
  [`ports/mql5/`](ports/mql5/) indicator and Expert Advisor.
- **Want to build on it locally?** Choose the Python, Rust, or Lean 4 port
  below depending on whether you need scripting ergonomics, compiled speed, or
  a formal executable specification.

A *port* is a reimplementation of a program in a different programming language
or for a different platform. The Python, Rust, Lean 4, and MQL5 ports under
[`ports/`](ports/) reproduce the original PineScript indicator's algorithm,
alongside the pinned PineScript source this repo keeps for review and parity
testing.

## Choose a port

| Port | Best for | Start here |
| --- | --- | --- |
| PineScript v6 | TradingView users who want the original chart indicator | [Open on TradingView](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/) |
| MQL5 | MetaTrader 5 users who want the indicator plus a Strategy Tester EA | [`ports/mql5/`](ports/mql5/) |
| Python | Research workflows, CSV exports, notebooks, and the quickest local smoke test | [`ports/python/`](ports/python/) |
| Rust | Fast local testing, dependency-light library usage, and performance-focused experiments | [`ports/rust/`](ports/rust/) |
| Lean 4 | An executable formal specification with stated and tested invariants | [`ports/lean/`](ports/lean/) |

Need another ecosystem? [Request a new port](https://github.com/artificial-intelligence-edge/lorentzian-classification/issues/new?template=port-request.yml).

## Fastest local smoke test

No accounts, API keys, or external data needed. This runs on a Coinbase
BTC/USD daily history committed to the repo and needs only Python 3. It is the
shortest way to verify the algorithm locally, not the only supported port. From
the repository root, compute the full result series:

```bash
PYTHONPATH=ports/python python3 -m lorentzian_classification run \
  tests/parity/baselines/pine_coinbase_btcusd_1d_limited_history.csv \
  --output /tmp/btcusd_daily_signals.csv
```

View the most recent signals:

```bash
column -s, -t < /tmp/btcusd_daily_signals.csv | tail -5
```

For Rust, Lean 4, cross-port parity, and bring-your-own TradingView CSV
workflows, see [`docs/examples.md`](docs/examples.md).

## Quick links

| Need | Link |
| --- | --- |
| Use the original indicator | [TradingView Editors' Picks](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/) |
| Use the MetaTrader 5 port | [`ports/mql5/`](ports/mql5/) |
| Read the settings reference | [ai-edge.io/docs](https://ai-edge.io/docs/indicators/lorentzian-classification/general-settings) |
| Explore optimizer studies | [AI Edge Optimizer](https://optimizer.ai-edge.io/studies) |
| Reproduce validation | [`docs/validation.md`](docs/validation.md) |
| Run the examples | [`docs/examples.md`](docs/examples.md) |
| Request another port | [New port request](https://github.com/artificial-intelligence-edge/lorentzian-classification/issues/new?template=port-request.yml) |

## What is in this repo

| Path | What it is |
| --- | --- |
| [`ports/pinescript/`](ports/pinescript/) | The SHA-pinned original TradingView indicator that everything else is checked against |
| [`ports/python/`](ports/python/) | Python CLI and library port; reproduces the gold baselines (the TradingView CSV exports we treat as ground truth) with exact signals and feature/kernel tolerance |
| [`ports/rust/`](ports/rust/) | Rust core library and CLI; bit-exact with the Python port |
| [`ports/lean/`](ports/lean/) | Lean 4 executable formal specification with proved structural invariants; byte-identical output to the Rust port |
| [`ports/mql5/`](ports/mql5/) | MetaTrader 5 indicator and Expert Advisor wrapper |
| [`docs/`](docs/) | Validation policy and copy-pasteable cross-port usage examples ([`docs/examples.md`](docs/examples.md)) |
| [`tests/`](tests/) | Parity fixtures, gold baselines, and the cross-port parity harness (`tests/parity/cross_port_parity.sh`) |

## Parity and validation

The PineScript source under [`ports/pinescript/`](ports/pinescript/) is the
algorithmic ground truth. The repository keeps TradingView export fixtures
under [`tests/parity/baselines/`](tests/parity/baselines/) and uses them to
verify the repo-runnable ports.

| Port | Validation status |
| --- | --- |
| PineScript | Manifest-pinned source files and libraries; fixture checks fail if the local reference drifts. |
| Python | Matches every tracked TradingView gold baseline under the shared feature/kernel/signal contract; strict non-default coverage is still expanding. |
| Rust | Recomputes the same baselines and is bit-exact with the Python port. |
| Lean 4 | Builds as an executable formal specification, passes theorem-named property tests, and is byte-identical to Rust on the committed baselines. |
| MQL5 | Ships as a MetaTrader 5 indicator plus EA; validation is documented in the port because it runs inside MT5 rather than the repo CSV harness. |

See [`docs/validation.md`](docs/validation.md) for the exact parity contract,
commands, known platform differences, and current coverage notes.

## Disclaimer

Lorentzian Classification is impersonal indicator software: it surfaces
classification signals computed from historical market data, and how you act on
them is your decision. It is a tool for knowledgeable traders, not a system
that replaces skill or judgment, and it does not provide financial advice or
personalized recommendations. Past performance does not guarantee future
results.

## License

Released under the [MIT License](LICENSE.md) — use it freely, just keep the
copyright notice. The only exception is the PineScript reference indicator and
its libraries under `ports/pinescript/`, which retain their original MPL-2.0
headers.
