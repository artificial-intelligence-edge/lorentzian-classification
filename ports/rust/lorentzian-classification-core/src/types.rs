//! Core data types: bars, settings, feature specifications, and result rows.
//!
//! These mirror the dataclasses in the Python reference port (`core.py`) so the
//! two implementations stay structurally aligned and bit-for-bit comparable.

use std::fmt;

/// Sentinel for a missing value, equivalent to Pine `na` / Python `math.nan`.
pub const MISSING: f64 = f64::NAN;

/// Returns `true` when `value` represents a missing (`NaN`) observation.
#[inline]
#[must_use]
pub fn is_missing(value: f64) -> bool {
    value.is_nan()
}

/// Returns `fallback` when `value` is missing, otherwise `value`
/// (equivalent to Pine `nz`).
#[inline]
#[must_use]
pub fn nz(value: f64, fallback: f64) -> f64 {
    if is_missing(value) {
        fallback
    } else {
        value
    }
}

/// A single OHLC price bar. `time` is preserved verbatim from the input feed.
#[derive(Debug, Clone, PartialEq)]
pub struct Bar {
    /// Bar timestamp, kept exactly as the input feed rendered it.
    pub time: String,
    /// Opening price of the bar.
    pub open: f64,
    /// Highest traded price of the bar.
    pub high: f64,
    /// Lowest traded price of the bar.
    pub low: f64,
    /// Closing price of the bar.
    pub close: f64,
}

/// The price series a feature or kernel reads from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Source {
    /// The bar open.
    Open,
    /// The bar high.
    High,
    /// The bar low.
    Low,
    /// The bar close (the Pine default).
    #[default]
    Close,
    /// `(high + low) / 2`.
    Hl2,
    /// `(high + low + close) / 3`.
    Hlc3,
    /// `(open + high + low + close) / 4`.
    Ohlc4,
}

impl Source {
    /// Parses a Pine-style source name (case-insensitive).
    ///
    /// # Errors
    /// Returns [`ParseError`] when `name` is not a recognized source.
    pub fn parse(name: &str) -> Result<Self, ParseError> {
        match name.to_ascii_lowercase().as_str() {
            "open" => Ok(Self::Open),
            "high" => Ok(Self::High),
            "low" => Ok(Self::Low),
            "close" => Ok(Self::Close),
            "hl2" => Ok(Self::Hl2),
            "hlc3" => Ok(Self::Hlc3),
            "ohlc4" => Ok(Self::Ohlc4),
            other => Err(ParseError::UnsupportedSource(other.to_string())),
        }
    }
}

/// The kind of normalized feature computed for a feature slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// Normalized RSI of the close.
    Rsi,
    /// Normalized WaveTrend of hlc3.
    Wt,
    /// Normalized CCI of the close.
    Cci,
    /// Normalized ADX of high/low/close.
    Adx,
}

impl FeatureKind {
    /// Parses a Pine feature-kind token (case-insensitive).
    ///
    /// # Errors
    /// Returns [`ParseError`] when `name` is not a recognized feature kind.
    pub fn parse(name: &str) -> Result<Self, ParseError> {
        match name.to_ascii_uppercase().as_str() {
            "RSI" => Ok(Self::Rsi),
            "WT" => Ok(Self::Wt),
            "CCI" => Ok(Self::Cci),
            "ADX" => Ok(Self::Adx),
            other => Err(ParseError::UnsupportedFeature(other.to_string())),
        }
    }
}

/// A feature slot specification: kind plus the two Pine smoothing parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureSpec {
    /// Which normalized indicator this slot computes.
    pub kind: FeatureKind,
    /// Pine "Parameter A": the primary lookback length of the indicator.
    pub param_a: i64,
    /// Pine "Parameter B": the secondary length (EMA smoothing for RSI/CCI,
    /// the average length for WT). ADX ignores it, matching the reference.
    pub param_b: i64,
}

impl FeatureSpec {
    /// Convenience constructor.
    #[must_use]
    pub const fn new(kind: FeatureKind, param_a: i64, param_b: i64) -> Self {
        Self {
            kind,
            param_a,
            param_b,
        }
    }

    /// Parses a `KIND:a:b` specification string (e.g. `"CCI:34:2"`).
    ///
    /// # Errors
    /// Returns [`ParseError`] when the string is malformed.
    pub fn parse(spec: &str) -> Result<Self, ParseError> {
        let mut parts = spec.split(':');
        let kind = parts
            .next()
            .ok_or_else(|| ParseError::MalformedFeatureSpec(spec.to_string()))?;
        let a = parts
            .next()
            .ok_or_else(|| ParseError::MalformedFeatureSpec(spec.to_string()))?;
        let b = parts
            .next()
            .ok_or_else(|| ParseError::MalformedFeatureSpec(spec.to_string()))?;
        if parts.next().is_some() {
            return Err(ParseError::MalformedFeatureSpec(spec.to_string()));
        }
        let parse_int = |s: &str| {
            s.trim()
                .parse::<i64>()
                .map_err(|_| ParseError::MalformedFeatureSpec(spec.to_string()))
        };
        Ok(Self::new(
            FeatureKind::parse(kind.trim())?,
            parse_int(a)?,
            parse_int(b)?,
        ))
    }
}

/// Errors produced while parsing user-supplied configuration values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The source name is not one of `open/high/low/close/hl2/hlc3/ohlc4`.
    UnsupportedSource(String),
    /// The feature kind is not one of `RSI/WT/CCI/ADX`.
    UnsupportedFeature(String),
    /// The feature spec is not a well-formed `KIND:a:b` triple.
    MalformedFeatureSpec(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSource(s) => write!(f, "unsupported source: {s}"),
            Self::UnsupportedFeature(s) => write!(f, "unsupported feature type: {s}"),
            Self::MalformedFeatureSpec(s) => write!(f, "malformed feature spec: {s}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Indicator settings, mirroring the Pine `input.*` declarations and the
/// Python `Settings` dataclass defaults exactly.
///
/// Every field maps to a named input in the PineScript reference (noted per
/// field), so a TradingView configuration can be transcribed directly.
/// [`Settings::default()`] reproduces the reference defaults; start from it and
/// override only what the chart changes:
///
/// ```
/// use lorentzian_classification_core::Settings;
///
/// let settings = Settings {
///     neighbors_count: 12,
///     max_bars_back: 1000,
///     use_adx_filter: true,
///     ..Settings::default()
/// };
/// assert_eq!(settings.feature_count, 5); // untouched defaults remain
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// Price series fed to the kernels and the four-bar training labels
    /// (Pine "Source").
    pub source: Source,
    /// Number of nearest neighbors summed into each prediction
    /// (Pine "Neighbors Count").
    pub neighbors_count: i64,
    /// Maximum lookback of the training window, in bars
    /// (Pine "Max Bars Back").
    pub max_bars_back: i64,
    /// How many of the five feature slots participate in the Lorentzian
    /// distance (Pine "Feature Count", 2 to 5).
    pub feature_count: usize,
    /// Pine "Color Compression" input. Carried for settings fidelity with the
    /// reference; the ported confidence gradient reproduces the default
    /// compression, so this field does not alter computed outputs.
    pub color_compression: i64,
    /// Extends the neighbor scan back to the first bar of the chart instead
    /// of starting at `max_bars_back_index`, picking up training patterns
    /// from the full history (Pine "Include Full History"). The gold
    /// baselines cover both modes.
    pub include_full_history: bool,
    /// Enables the ATR(1) > ATR(10) volatility gate on signals
    /// (Pine "Use Volatility Filter").
    pub use_volatility_filter: bool,
    /// Enables the KLMF regime-slope gate on signals
    /// (Pine "Use Regime Filter").
    pub use_regime_filter: bool,
    /// Enables the ADX(14) trend-strength gate on signals
    /// (Pine "Use ADX Filter").
    pub use_adx_filter: bool,
    /// Normalized-slope threshold for the regime filter
    /// (Pine "Threshold" under the regime filter).
    pub regime_threshold: f64,
    /// Minimum ADX value required when the ADX filter is enabled
    /// (Pine "Threshold" under the ADX filter).
    pub adx_threshold: i64,
    /// Requires close above/below the EMA for long/short entries
    /// (Pine "Use EMA Filter").
    pub use_ema_filter: bool,
    /// EMA length for the EMA trend filter (Pine "Period").
    pub ema_period: i64,
    /// Requires close above/below the SMA for long/short entries
    /// (Pine "Use SMA Filter").
    pub use_sma_filter: bool,
    /// SMA length for the SMA trend filter (Pine "Period").
    pub sma_period: i64,
    /// Gates entries on the kernel trend direction
    /// (Pine "Trade with Kernel").
    pub use_kernel_filter: bool,
    /// Uses the Gaussian/rational-quadratic crossover instead of the kernel
    /// rate of change for trend detection
    /// (Pine "Enhance Kernel Smoothing").
    pub use_kernel_smoothing: bool,
    /// Exits on kernel reversals instead of the fixed four-bar rule; only
    /// active while the EMA/SMA filters and kernel smoothing are all off
    /// (Pine "Use Dynamic Exits").
    pub use_dynamic_exits: bool,
    /// Emits the displayed exit markers [`ResultRow::stop_buy`] and
    /// [`ResultRow::stop_sell`], which fire exactly four bars after an entry
    /// (Pine "Show Default Exits").
    pub show_exits: bool,
    /// Trade statistics price basis: the configured [`Self::source`] when
    /// `true`, otherwise the `(high + low + 2 * open) / 4` estimate
    /// (Pine "Use Worst Case Estimates").
    pub use_worst_case: bool,
    /// Kernel lookback window `h` (Pine "Lookback Window").
    pub kernel_h: i64,
    /// Kernel relative weighting `r` for the rational quadratic kernel
    /// (Pine "Relative Weighting").
    pub kernel_r: f64,
    /// Bar index where kernel regression starts (Pine "Regression Level").
    pub kernel_x: i64,
    /// Lag between the rational quadratic and Gaussian kernel estimates,
    /// used for crossover detection (Pine "Lag").
    pub kernel_lag: i64,
    /// Colors the kernel plot column; when `false` the kernel plot color is
    /// fully transparent (Pine "Show Kernel Estimate").
    pub show_kernel_estimate: bool,
    /// Emits gradient bar colors in [`ResultRow::bar_color`]
    /// (Pine "Show Bar Colors").
    pub show_bar_colors: bool,
    /// Emits the prediction label color in
    /// [`ResultRow::prediction_label_color`] (Pine "Show Bar Prediction Values").
    pub show_bar_predictions: bool,
    /// Offsets prediction labels by ATR(1) instead of the fixed
    /// `hl2`-scaled offset (Pine "Use ATR Offset").
    pub use_atr_offset: bool,
    /// Vertical offset factor for prediction labels when
    /// [`Self::use_atr_offset`] is off (Pine "Bar Prediction Offset").
    pub bar_predictions_offset: f64,
    /// Shades prediction and bar colors by prediction magnitude; solid
    /// colors when `false` (the Pine confidence gradient).
    pub use_confidence_gradient: bool,
    /// Emits the trade statistics visibility flag and header
    /// (Pine "Show Trade Stats").
    pub show_trade_stats: bool,
    /// Feature slot 1 (Pine "Feature 1" with parameters A/B).
    pub f1: FeatureSpec,
    /// Feature slot 2 (Pine "Feature 2" with parameters A/B).
    pub f2: FeatureSpec,
    /// Feature slot 3 (Pine "Feature 3" with parameters A/B).
    pub f3: FeatureSpec,
    /// Feature slot 4 (Pine "Feature 4" with parameters A/B).
    pub f4: FeatureSpec,
    /// Feature slot 5 (Pine "Feature 5" with parameters A/B).
    pub f5: FeatureSpec,
}

impl Default for Settings {
    fn default() -> Self {
        use FeatureKind::{Adx, Cci, Rsi, Wt};
        Self {
            source: Source::Close,
            neighbors_count: 8,
            max_bars_back: 2000,
            feature_count: 5,
            color_compression: 1,
            include_full_history: false,
            use_volatility_filter: true,
            use_regime_filter: true,
            use_adx_filter: false,
            regime_threshold: -0.1,
            adx_threshold: 20,
            use_ema_filter: false,
            ema_period: 200,
            use_sma_filter: false,
            sma_period: 200,
            use_kernel_filter: true,
            use_kernel_smoothing: false,
            use_dynamic_exits: false,
            show_exits: false,
            use_worst_case: false,
            kernel_h: 8,
            kernel_r: 8.0,
            kernel_x: 25,
            kernel_lag: 2,
            show_kernel_estimate: true,
            show_bar_colors: true,
            show_bar_predictions: true,
            use_atr_offset: true,
            bar_predictions_offset: 0.0,
            use_confidence_gradient: true,
            show_trade_stats: true,
            f1: FeatureSpec::new(Rsi, 14, 1),
            f2: FeatureSpec::new(Wt, 10, 11),
            f3: FeatureSpec::new(Cci, 20, 1),
            f4: FeatureSpec::new(Adx, 20, 2),
            f5: FeatureSpec::new(Rsi, 9, 1),
        }
    }
}

impl Settings {
    /// Returns the five feature slots in order.
    #[must_use]
    pub fn features(&self) -> [FeatureSpec; 5] {
        [self.f1, self.f2, self.f3, self.f4, self.f5]
    }
}

/// A fully computed per-bar result row, mirroring the Python `ResultRow`.
///
/// One row is produced per input bar; the field order matches the 40-column
/// output schema in [`RESULT_FIELDNAMES`]. Colors are Pine-style
/// `"#RRGGBB@transparency"` strings (see [`crate::display`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ResultRow {
    /// The input bar this row was computed from.
    pub bar: Bar,
    /// Normalized value of feature slot 1 ([`Settings::f1`]).
    pub f1: f64,
    /// Normalized value of feature slot 2 ([`Settings::f2`]).
    pub f2: f64,
    /// Normalized value of feature slot 3 ([`Settings::f3`]).
    pub f3: f64,
    /// Normalized value of feature slot 4 ([`Settings::f4`]).
    pub f4: f64,
    /// Normalized value of feature slot 5 ([`Settings::f5`]).
    pub f5: f64,
    /// Rational quadratic kernel regression estimate (Pine `yhat1`).
    pub kernel: f64,
    /// Summed neighbor label, in
    /// `[-neighbors_count, neighbors_count]`; `0` during warmup.
    pub prediction: i64,
    /// Persisted signal after filtering: `1` long, `-1` short, `0` before
    /// the first filtered prediction.
    pub direction: i64,
    /// `true` on bars where a long entry starts (Pine `startLongTrade`).
    pub buy: bool,
    /// `true` on bars where a short entry starts (Pine `startShortTrade`).
    pub sell: bool,
    /// `true` when the long exit condition fires (four-bar rule, or the
    /// kernel reversal under dynamic exits).
    pub exit_buy: bool,
    /// `true` when the short exit condition fires.
    pub exit_sell: bool,
    /// Displayed long exit marker: [`Self::exit_buy`] gated by
    /// [`Settings::show_exits`].
    pub stop_buy: bool,
    /// Displayed short exit marker: [`Self::exit_sell`] gated by
    /// [`Settings::show_exits`].
    pub stop_sell: bool,
    /// Encoded event stream: `1` start long, `2` end long, `-1` start short,
    /// `-2` end short, `0` no event.
    pub backtest_stream: i64,
    /// Pine "Open Long" `alertcondition` trigger.
    pub open_long_alert: bool,
    /// Pine "Close Long" `alertcondition` trigger.
    pub close_long_alert: bool,
    /// Pine "Open Short" `alertcondition` trigger.
    pub open_short_alert: bool,
    /// Pine "Close Short" `alertcondition` trigger.
    pub close_short_alert: bool,
    /// Pine "Open Position" trigger: either entry started on this bar.
    pub open_position_alert: bool,
    /// Pine "Close Position" trigger: either exit fired on this bar.
    pub close_position_alert: bool,
    /// Kernel turned bullish (color change, or crossover under smoothing).
    pub kernel_bullish_alert: bool,
    /// Kernel turned bearish (color change, or crossover under smoothing).
    pub kernel_bearish_alert: bool,
    /// Color of the kernel estimate plot for this bar.
    pub kernel_plot_color: String,
    /// The prediction rendered as the on-chart label text.
    pub prediction_label: String,
    /// Chart y-coordinate for the prediction label (ATR- or offset-based,
    /// per [`Settings::use_atr_offset`]).
    pub prediction_label_y: f64,
    /// Prediction label color; empty when labels are hidden.
    pub prediction_label_color: String,
    /// Bar color; empty when bar coloring is disabled.
    pub bar_color: String,
    /// Mirrors [`Settings::show_trade_stats`].
    pub trade_stats_visible: bool,
    /// Header text of the Pine trade statistics table.
    pub trade_stats_header: String,
    /// Running count of winning trades.
    pub total_wins: i64,
    /// Running count of losing trades.
    pub total_losses: i64,
    /// Running count of signals that flipped within four bars of the
    /// previous flip (a choppiness indication).
    pub total_early_signal_flips: i64,
    /// Running count of closed trades (`total_wins + total_losses`).
    pub total_trades: i64,
    /// `total_wins / total_trades`; NaN before the first closed trade.
    pub win_loss_ratio: f64,
    /// `total_wins / total_losses` as shown in the Pine stats table;
    /// NaN while there are no losses.
    pub table_wl_ratio: f64,
    /// `total_wins / (total_wins + total_losses)`; NaN before the first
    /// closed trade.
    pub win_rate: f64,
}

/// Full output schema, identical to Python `RESULT_FIELDNAMES`.
pub const RESULT_FIELDNAMES: [&str; 40] = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "F1_RSI",
    "F2_WT",
    "F3_CCI",
    "F4_ADX",
    "F5_RSI9",
    "Kernel Regression Estimate",
    "Prediction",
    "Direction",
    "Buy",
    "Sell",
    "StopBuy",
    "StopSell",
    "Backtest Stream",
    "Open Long Alert",
    "Close Long Alert",
    "Open Short Alert",
    "Close Short Alert",
    "Open Position Alert",
    "Close Position Alert",
    "Kernel Bullish Alert",
    "Kernel Bearish Alert",
    "Kernel Plot Color",
    "Prediction Label",
    "Prediction Label Y",
    "Prediction Label Color",
    "Bar Color",
    "Trade Stats Visible",
    "Trade Stats Header",
    "Total Wins",
    "Total Losses",
    "Total Early Signal Flips",
    "Total Trades",
    "Win Loss Ratio",
    "Table WL Ratio",
    "Win Rate",
];
