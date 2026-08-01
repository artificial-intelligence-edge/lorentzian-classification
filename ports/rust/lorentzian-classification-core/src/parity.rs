//! Parity comparison against TradingView/Pine export CSVs.
//!
//! This mirrors `parity_summary` in the Python reference CLI: the comparison
//! window starts at `max_bars_back_index` (0 under full history), numeric
//! feature/kernel columns must agree within a tolerance (NaN on both sides is
//! skipped), and prediction/direction/buy/sell/stop columns must match exactly.

use std::path::Path;

use crate::csv_io::{column_index, detect_price_scale, CsvError};
use crate::types::{is_missing, Bar, ResultRow, Settings};

/// The expected per-bar values parsed from a Pine export row.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedRow {
    /// Normalized feature slots `F1..F5` (NaN when the export cell was empty).
    pub features: [f64; 5],
    /// Kernel regression estimate (NaN when empty).
    pub kernel: f64,
    /// Exported integer prediction.
    pub prediction: i64,
    /// Exported integer direction.
    pub direction: i64,
    /// Exported long-entry flag.
    pub buy: bool,
    /// Exported short-entry flag.
    pub sell: bool,
    /// Exported displayed long-exit flag.
    pub stop_buy: bool,
    /// Exported displayed short-exit flag.
    pub stop_sell: bool,
}

/// A single column disagreement found during comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    /// Row index (0-based) where the disagreement occurred.
    pub index: usize,
    /// Logical column name (`f1`..`f5`, `kernel`, `prediction`, …).
    pub column: String,
    /// Rendered expected (Pine) value.
    pub expected: String,
    /// Rendered computed value.
    pub actual: String,
}

impl Mismatch {
    fn new(index: usize, column: impl Into<String>, expected: String, actual: String) -> Self {
        Self {
            index,
            column: column.into(),
            expected,
            actual,
        }
    }
}

/// Aggregate result of a parity comparison.
#[derive(Debug, Clone)]
pub struct ParitySummary {
    /// `true` when no mismatches were found.
    pub pass: bool,
    /// Number of rows whose signal columns were compared (rows from
    /// `max_bars_back_index` onward).
    pub compared: usize,
    /// First row index where Pine emits predictions:
    /// `max(0, last_bar_index - max_bars_back)`.
    pub max_bars_back_index: i64,
    /// Largest absolute feature difference observed across compared cells.
    pub max_feature_diff: f64,
    /// Largest absolute kernel-estimate difference observed.
    pub max_kernel_diff: f64,
    /// Every column disagreement found, in row order.
    pub mismatches: Vec<Mismatch>,
}

/// Pine `nz`/`parse_bool` semantics: empty / `0` / `false` / `na` / `nan` are
/// falsey; any other non-empty token is truthy.
fn parse_bool(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    !matches!(normalized.as_str(), "" | "0" | "false" | "na" | "nan")
}

fn parse_float_cell(value: &str) -> f64 {
    let v = value.trim();
    if v.is_empty() {
        f64::NAN
    } else {
        v.parse::<f64>().unwrap_or(f64::NAN)
    }
}

fn parse_int_cell(value: &str) -> i64 {
    let v = value.trim();
    if v.is_empty() {
        0
    } else {
        // Mirrors Python `int(float(x))`: parse as float then truncate.
        v.parse::<f64>().map_or(0, |f| f.trunc() as i64)
    }
}

fn header_index(header: &csv::StringRecord, name: &str) -> Option<usize> {
    header
        .iter()
        .position(|c| c.trim().eq_ignore_ascii_case(name))
}

/// Finds a feature column by exact name, then by `F{slot}` prefix
/// (mirrors `get_feature_slot` in the Python reader).
fn feature_index(header: &csv::StringRecord, slot: usize) -> Option<usize> {
    let prefix = format!("F{slot}");
    header
        .iter()
        .position(|c| c.trim().to_ascii_uppercase().starts_with(&prefix))
}

/// Reads OHLC bars, the expected comparison columns, and the derived ADX price
/// scale from a Pine export CSV.
///
/// # Errors
/// Returns [`CsvError`] when the file cannot be read, is empty, or lacks a
/// required OHLC column.
pub fn read_pine_export(path: &Path) -> Result<(Vec<Bar>, Vec<ExpectedRow>, f64), CsvError> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_path(path)?;
    let header = reader.headers()?.clone();
    if header.is_empty() {
        return Err(CsvError::Empty);
    }

    let time_i = column_index(&header, "time")?;
    let open_i = column_index(&header, "open")?;
    let high_i = column_index(&header, "high")?;
    let low_i = column_index(&header, "low")?;
    let close_i = column_index(&header, "close")?;
    let feat_i: [Option<usize>; 5] = [
        feature_index(&header, 1),
        feature_index(&header, 2),
        feature_index(&header, 3),
        feature_index(&header, 4),
        feature_index(&header, 5),
    ];
    let kernel_i = header_index(&header, "Kernel Regression Estimate");
    let prediction_i = header_index(&header, "Prediction");
    let direction_i = header_index(&header, "Direction");
    let buy_i = header_index(&header, "Buy");
    let sell_i = header_index(&header, "Sell");
    let stop_buy_i = header_index(&header, "StopBuy");
    let stop_sell_i = header_index(&header, "StopSell");

    let mut bars = Vec::new();
    let mut expected = Vec::new();
    let mut ohlc_cells: Vec<[String; 4]> = Vec::new();

    for record in reader.records() {
        let record = record?;
        if record.iter().all(|cell| cell.trim().is_empty()) {
            continue;
        }
        let line_no = record
            .position()
            .map_or(0, |position| position.line() as usize);
        let cell = |i: usize| record.get(i).map_or("", str::trim);
        let parse_ohlc = |i: usize, col: &'static str| {
            cell(i).parse::<f64>().map_err(|_| CsvError::ParseField {
                line: line_no,
                column: col,
                value: cell(i).to_string(),
            })
        };
        bars.push(Bar {
            time: cell(time_i).to_string(),
            open: parse_ohlc(open_i, "open")?,
            high: parse_ohlc(high_i, "high")?,
            low: parse_ohlc(low_i, "low")?,
            close: parse_ohlc(close_i, "close")?,
        });
        ohlc_cells.push([
            cell(open_i).to_string(),
            cell(high_i).to_string(),
            cell(low_i).to_string(),
            cell(close_i).to_string(),
        ]);
        let feat = |slot: usize| feat_i[slot].map_or(f64::NAN, |i| parse_float_cell(cell(i)));
        expected.push(ExpectedRow {
            features: [feat(0), feat(1), feat(2), feat(3), feat(4)],
            kernel: kernel_i.map_or(f64::NAN, |i| parse_float_cell(cell(i))),
            prediction: prediction_i.map_or(0, |i| parse_int_cell(cell(i))),
            direction: direction_i.map_or(0, |i| parse_int_cell(cell(i))),
            buy: buy_i.is_some_and(|i| parse_bool(cell(i))),
            sell: sell_i.is_some_and(|i| parse_bool(cell(i))),
            stop_buy: stop_buy_i.is_some_and(|i| parse_bool(cell(i))),
            stop_sell: stop_sell_i.is_some_and(|i| parse_bool(cell(i))),
        });
    }

    let price_scale = detect_price_scale(&ohlc_cells);
    Ok((bars, expected, price_scale))
}

/// Compares computed `results` against `expected` Pine values.
///
/// Mirrors the Python `parity_summary` exactly:
///
/// * Features `F1..F5` and the kernel are compared on **every** row, skipping a
///   column whenever the *expected* (Pine) value is NaN (Pine had no value
///   there). When the expected value is present, the computed value must be
///   within `tolerance`.
/// * Prediction, direction, buy, sell, and the two stop signals are compared
///   only from `max_bars_back_index = max(0, last_bar_index - max_bars_back)`
///   onward (the first bar Pine emits a prediction for). This index is *not*
///   adjusted for `include_full_history`, matching the reference comparator.
#[must_use]
pub fn parity_summary(
    expected: &[ExpectedRow],
    results: &[ResultRow],
    tolerance: f64,
    settings: &Settings,
) -> ParitySummary {
    let last_bar_index = expected.len() as i64 - 1;
    let max_bars_back_index = if last_bar_index >= settings.max_bars_back {
        last_bar_index - settings.max_bars_back
    } else {
        0
    };

    let mut mismatches = Vec::new();
    let mut compared = 0_usize;
    let mut max_feature_diff = 0.0_f64;
    let mut max_kernel_diff = 0.0_f64;

    for (index, (exp, res)) in expected.iter().zip(results.iter()).enumerate() {
        // Features and kernel are compared on every row (warmup included),
        // skipping columns where Pine has no value.
        let res_features = [res.f1, res.f2, res.f3, res.f4, res.f5];
        for (slot, (&e, &a)) in exp.features.iter().zip(res_features.iter()).enumerate() {
            if is_missing(e) {
                continue;
            }
            let column = || format!("f{}", slot + 1);
            if is_missing(a) {
                mismatches.push(Mismatch::new(index, column(), fmt(e), fmt(a)));
                continue;
            }
            let diff = (e - a).abs();
            max_feature_diff = max_feature_diff.max(diff);
            if diff > tolerance {
                mismatches.push(Mismatch::new(index, column(), fmt(e), fmt(a)));
            }
        }
        if !is_missing(exp.kernel) {
            if is_missing(res.kernel) {
                mismatches.push(Mismatch::new(
                    index,
                    "kernel",
                    fmt(exp.kernel),
                    fmt(res.kernel),
                ));
            } else {
                let diff = (exp.kernel - res.kernel).abs();
                max_kernel_diff = max_kernel_diff.max(diff);
                if diff > tolerance {
                    mismatches.push(Mismatch::new(
                        index,
                        "kernel",
                        fmt(exp.kernel),
                        fmt(res.kernel),
                    ));
                }
            }
        }

        // Signals are only meaningful from max_bars_back_index onward.
        if (index as i64) < max_bars_back_index {
            continue;
        }
        compared += 1;
        check_int(
            &mut mismatches,
            index,
            "prediction",
            exp.prediction,
            res.prediction,
        );
        check_int(
            &mut mismatches,
            index,
            "direction",
            exp.direction,
            res.direction,
        );
        check_bool(&mut mismatches, index, "buy", exp.buy, res.buy);
        check_bool(&mut mismatches, index, "sell", exp.sell, res.sell);
        check_bool(
            &mut mismatches,
            index,
            "stop_buy",
            exp.stop_buy,
            res.stop_buy,
        );
        check_bool(
            &mut mismatches,
            index,
            "stop_sell",
            exp.stop_sell,
            res.stop_sell,
        );
    }

    ParitySummary {
        pass: mismatches.is_empty(),
        compared,
        max_bars_back_index,
        max_feature_diff,
        max_kernel_diff,
        mismatches,
    }
}

fn fmt(value: f64) -> String {
    if value.is_nan() {
        "na".to_string()
    } else {
        format!("{value}")
    }
}

fn check_int(out: &mut Vec<Mismatch>, index: usize, column: &str, expected: i64, actual: i64) {
    if expected != actual {
        out.push(Mismatch::new(
            index,
            column,
            expected.to_string(),
            actual.to_string(),
        ));
    }
}

fn check_bool(out: &mut Vec<Mismatch>, index: usize, column: &str, expected: bool, actual: bool) {
    if expected != actual {
        out.push(Mismatch::new(
            index,
            column,
            expected.to_string(),
            actual.to_string(),
        ));
    }
}
