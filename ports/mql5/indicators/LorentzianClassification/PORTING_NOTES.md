# Porting Notes: PineScript → MQL5

This document captures the PineScript equivalents and compatibility notes that
guided the MQL5 implementation of the Lorentzian Classification indicator. It
exists so the source files themselves can stay clean for open-source consumers
while the line-by-line correspondence with jdehorty's original PineScript
remains available for anyone auditing the port or making parity changes.

Source files:

- Original PineScript:
  `PineScript/Indicators/LorentzianClassification.pine`
  Copyright (c) jdehorty, Mozilla Public License 2.0
  (https://mozilla.org/MPL/2.0/)
- PineScript libraries referenced by the indicator:
  - `jdehorty/MLExtensions/2`
  - `jdehorty/KernelFunctions/2`
- MQL5 port (this folder):
  - `LorentzianClassification.mq5`: indicator entry point
  - `Include/MLFeatures.mqh`: feature-engineering primitives (normalized
    RSI, WaveTrend, CCI, ADX, EMA, SMA, RMA, ATR, Wilder smoothing,
    min-max normalize, linear rescale)
  - `Include/KernelFunctions.mqh`: Rational Quadratic and Gaussian kernel
    regressions
  - `Include/Filters.mqh`: volatility / regime / ADX / EMA / SMA filters
  - `Include/ANN.mqh`: greedy approximate nearest neighbors classifier
  - `Include/Backtest.mqh`: real-time trade stats
  - `Include/SignalAlerts.mqh`: confirmed entry-alert formatting,
    deduplication, terminal alerts, and optional mobile push delivery

---

## Language-level translations

| PineScript concept | MQL5 equivalent in this port |
|---|---|
| `na` (missing value sentinel) | `LC_EMPTY` (`= EMPTY_VALUE`) in `MLFeatures.mqh` |
| `nz(x)` / `nz(x, fallback)` | `NZ(x)` / `NZ(x, fallback)` helper |
| `ta.ema(src, n)` | `CalcEMA(src, n, out, begin, total)`: SMA-seeded alpha recurrence, alpha = 2/(n+1) |
| `ta.sma(src, n)` | `CalcSMA(src, n, out, begin, total)` |
| `ta.rma(src, n)` | `CalcRMA(src, n, out, begin, total)`: SMA-seeded, alpha = 1/n |
| `ta.atr(n)` | `CalcATR(high, low, close, n, out, tr, begin, total)` |
| `ta.rsi(src, n)` | `CalcRSI(src, n, out, ...)` |
| `ta.cci(src, n)` | `CalcCCI(src, n, out, sma, begin, total)` |
| Wilder directional smoothing (no SMA seed) | `CalcWilderSmooth(src, n, out, ...)` |
| Min-max normalize via running extremes | `ApplyNormalize(src, outMin, outMax, normState)` |
| Fixed-range rescale | `ApplyRescale(src, oldMin, oldMax, newMin, newMax)` |
| `var x = ...` persistent series | MQL5 `static` locals or members of `ANNState` / `FilterState` that are reset only when `prev_calculated == 0` |
| `src[4] < src[0]` (backward-indexed lookback) | Forward-indexed `src[i - 4] < src[i]` inside the bar loop |
| `display=display.none` hidden plot | MQL5 `DRAW_NONE` buffers (e.g. `DirectionBuf`, `PredictionBuf`) |

### `na` propagation semantics

In PineScript, any arithmetic involving `na` yields `na`. A comparison
involving `na` evaluates to `false`. The MQL5 port preserves these semantics
by propagating `LC_EMPTY` through dependent calculations and by explicitly
guarding comparisons against `LC_EMPTY`.

Example: in the Lorentzian distance, any `na` operand in `math.log(1 + ...)`
makes the whole distance `na`, so `d >= lastDistance` is falsy and the
candidate is silently skipped. `GetLorentzianDistance` emulates this by
returning `-DBL_MAX` when any active feature (current or history) is
`LC_EMPTY`, guaranteeing the caller's `d >= lastDistance` test fails for any
typical `lastDistance` (including the initial value of `-1.0`).

### Bar-0 `nz()` defaults

PineScript frequently writes `nz(close[1])` to read the previous bar's close
with a default of `0` on bar 0. The MQL5 port mirrors this by using a
`(i > 0) ? x[i - 1] : 0` pattern in the same places (ATR, ADX, ADX filter).

### Truthiness of integers

PineScript treats a nonzero integer as truthy. `d >= lastDistance and i%4`
relies on this: when `i % 4 == 0` the expression is falsy. The MQL5 port
makes this explicit as `(i % 4) != 0`.

### `array.from(_src)` in kernel loops

The PineScript kernel functions loop
`for i = 0 to array.size(array.from(_src)) + startAtBar`. Since
`array.from(_src)` is a one-element array, `array.size = 1`, giving an
effective loop range of `0..(1 + startAtBar)`. The MQL5 port writes this
directly as `int limit = 1 + startAtBar;` and then
`for(int i = 0; i <= limit && i <= barIndex; i++)`.

---

## File-level correspondence

### `LorentzianClassification.mq5` ↔ `LorentzianClassification.pine`

Pine line numbers below refer to the v1 source this port was written against,
archived at `ports/pinescript/archive/lorentzian-classification-v1-020823-2301.pine`
(562 lines). They are not line numbers in `ports/pinescript/lorentzian-classification-v2.pine`,
where the same sections sit lower in the file.

| Pine section | Pine lines | MQL5 location |
|---|---|---|
| Background (Euclidean vs Lorentzian) | 10-106 | File header block in `.mq5` |
| Custom types | 115-163 | Represented as C-style `struct`s in include files (`ANNState`, `FilterState`, `FeatureWork`, etc.) |
| Helper functions (`series_from`, `get_lorentzian_distance`) | 168-190 | `CalcFeature()` (MLFeatures.mqh), `GetLorentzianDistance()` (ANN.mqh) |
| Settings inputs | 197-211, 232-315 | `input` declarations at the top of `.mq5` |
| FeatureArrays push | 258-279 | `ANNPushBar()` called inside the main bar loop |
| Next-bar classification & training label | 317-334 | Training-label comment block inside the bar loop, then `ANNPushBar(..., trainLabel)` |
| Core ML logic (ANN loop) | 335-398 | `RunANN()` in ANN.mqh |
| Prediction filters (`filter_all`) | 400-408 | `filterAll = filtVol && filtRegime && filtAdx` inside the bar loop |
| Bar-count / fractal filters (isDifferentSignalType, isEarlySignalFlip, barsHeld) | 410-424 | `isDiffSignalType`, `isEarlyFlip`, `barsHeld` bookkeeping in the bar loop |
| Kernel regression filters | 426-458 | "Kernel Regression Filters" block inside the bar loop |
| Entries & exits | 459-484 | "Entries and Exits" block: `startLong`, `startShort`, `endLongStrict`, `endLongDynamic`, etc. |
| Plotting labels | 486-494 | Signal buffer fills (`BuyBuf[i] = low[i]`, etc.) and post-loop label objects |
| Alerts | 496-512 | Buy/Sell entry alerts use `DispatchConfirmedSignalAlert()` beside the signal-buffer writes; exit and kernel-color alerts remain a documented divergence below |
| Display signals & bar coloring | 514-526 | "Display Signals" block using the color palette built in `OnInit` |
| Backtesting stream & stats | 528-562 | `UpdateBacktest()` / `DrawStatsTable()` |

### `Include/MLFeatures.mqh` ↔ `MLExtensions.pine`

| Pine function | MQL5 function |
|---|---|
| `MLExtensions.n_rsi(src, n1, n2)` | `CalcNormalizedRSI(src, n1, n2, out, ...)` |
| `MLExtensions.n_wt(hlc3, n1, n2)` | `CalcWaveTrend(src, n1, n2, out, ..., normState)` |
| `MLExtensions.n_cci(src, n1, n2)` | `CalcNormalizedCCI(src, n1, n2, out, ..., normState)` |
| `MLExtensions.n_adx(high, low, close, n)` | `CalcADX(high, low, close, n, out, ...)` (output rescaled to [0,1]) |
| `MLExtensions.rescale(x, oldMin, oldMax, newMin, newMax)` | `ApplyRescale(...)` |
| `MLExtensions.normalize(x, outMin, outMax)` | `ApplyNormalize(x, outMin, outMax, normState)` |
| `MLExtensions.filter_volatility(min, max, useFilter)` | `GetVolatilityFilter(fs, i, useFilter)` |
| `MLExtensions.regime_filter(ohlc4, threshold, useFilter)` | `GetRegimeFilter(fs, i, threshold, useFilter)` with state computed in `CalcRegimeFilter` |
| `MLExtensions.filter_adx(src, n, threshold, useFilter)` | `GetADXFilter(fs, i, threshold, useFilter)` with state in `CalcADXFilter` |
| `MLExtensions.backtest(...)` | `UpdateBacktest()` / `DrawStatsTable()` |

Numerical parity: the math inside each of the `Calc*` helpers mirrors the
PineScript library exactly, including seeding and warmup rules. The
boundary conditions (bar 0 behaviour, `na` treatment) are documented
inline where they matter.

### `Include/KernelFunctions.mqh` ↔ `KernelFunctions.pine`

| Pine function | MQL5 function |
|---|---|
| `kernels.rationalQuadratic(src, h, r, x)` | `KernelRationalQuadratic(src, barIndex, lookback, relativeWeight, startAtBar)` |
| `kernels.gaussian(src, h, x)` | `KernelGaussian(src, barIndex, lookback, startAtBar)` |

Semantics preserved:
- Non-repainting: the MQL5 versions only read `src[barIndex - i]` for `i >= 0`.
- Loop range: `0..(1 + startAtBar)`, clamped to available history.
- Edge fallback: if `cumulativeWeight <= 0` (no usable history), return
  `src[barIndex]` instead of dividing by zero.

### `Include/Filters.mqh`

- **Volatility filter**: ATR(1) > ATR(10) (periods fixed in the call site).
  During warmup (ATR values still `LC_EMPTY`) the filter returns `true`
  so early bars are not suppressed.
- **Regime filter**: ports `MLExtensions.regime_filter`, a Kaufman-style
  adaptive slope. State variables `value1`, `value2`, `KLMF`, `absSlope`,
  `emaAbsSlope(200)` are persisted in `FilterState` and computed
  bar-by-bar because the recurrence is order-dependent.
- **ADX filter**: port of `MLExtensions.filter_adx` with a hardcoded
  period 14 at the call site. Computed independently from `FEATURE_ADX`
  (which is additionally rescaled to [0,1] for ML consumption).
- **EMA / SMA trend filters**: classical `close > MA` / `close < MA`
  gates; used at the entry stage, not the signal-flip gate.

### `Include/ANN.mqh`

Ports the core ML loop and state. Persistent queues (`distances`,
`predictions`) correspond to `var` arrays in the original PineScript; they
accumulate over the full history so that the neighbor pool is not reset
each bar. `lastDistance` is intentionally NOT persistent (matches
PineScript -- there is no `var` on it) and resets to `-1` every call.

The loop bound in the MQL5 port is `sizeLoop = min(maxBarsBack - 1, dataSize - 1)`,
equivalent to the PineScript
`size = min(maxBarsBack - 1, array.size(y_train_array) - 1); sizeLoop = min(maxBarsBack - 1, size)`.

### `Include/Backtest.mqh`

Ports `MLExtensions.backtest(...)`. The original PineScript additionally
emits a hidden `backTestStream` plot with codes `+1` (startLong),
`+2` (endLong), `-1` (startShort), `-2` (endShort) for downstream adapters.
The MQL5 port does not re-expose that stream as a separate indicator
buffer; chart consumers read the `BuyBuf` / `SellBuf` / `ExitBuyBuf` /
`ExitSellBuf` outputs directly, which carry the same information.

---

## Alert delivery semantics

The PineScript reference exposes entry alert conditions for new long and short
positions. The MQL5 port maps those entry events to `Alert()` and optionally
`SendNotification()` without changing the signal conditions or buffer values.

The alert path adds platform-specific safeguards:

- Both delivery settings are disabled by default.
- Only the newest fully closed bar is eligible for delivery.
- A first calculation or any full recalculation emits no historical alerts.
- Buy and Sell events are deduplicated independently by signal-bar timestamp.
- Push configuration or delivery failures are written to the Journal and do
  not interrupt indicator calculation.
- MetaTrader does not deliver terminal or push alerts in Strategy Tester, so
  delivery must be smoke-tested on a live, demo, or custom-symbol chart.

---

## Divergences from the original

The port is faithful to the algorithm but has two platform-level differences
worth noting:

1. **Bar-colour gradient midpoint.** PineScript's `color.from_gradient`
   fades weak predictions toward neutral gray `#787b86`. The MQL5 palette
   in `OnInit` fades toward white instead, so low-magnitude predictions
   appear washed out rather than greyed out. The saturated endpoints
   (`#009988` teal and `#CC3311` red) are identical.
2. **Entry alerts only.** Version 1.01 provides terminal and optional push
   delivery for confirmed Buy and Sell entries. PineScript's close-position
   and kernel color-change alert conditions are not exposed as MQL5 alert
   controls.

---

## Parity checklist for future changes

When updating either side to stay in sync, the following should be
verified in lockstep:

- Feature order and defaults (RSI/WT/CCI/ADX/RSI with periods 14/10-11/20-1/20-2/9-1).
- Training label formula: `src[i-4] < src[i] ? short : src[i-4] > src[i] ? long : neutral`.
- ANN loop: `(i % 4) != 0` gating, 75th-percentile `lastDistance` bump,
  FIFO shift when the queue exceeds `neighborsCount`.
- Filter combination: `filterAll = vol AND regime AND adx`; EMA/SMA are
  entry-level only.
- Kernel parameters: `yhat1 = rationalQuadratic(src, h, r, x)`,
  `yhat2 = gaussian(src, h - lag, x)`.
- Exit-mode selection: dynamic exits are only valid when EMA, SMA, and
  kernel smoothing are all disabled.
