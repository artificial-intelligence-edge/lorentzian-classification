//! Minimal TradingView/Pine OHLC CSV reader.
//!
//! Reads the `time,open,high,low,close,...` export schema and derives the
//! decimal `price_scale` used for ADX quantization, matching
//! `read_tradingview_csv`/`detect_price_scale` in the Python reference.
//! Parsing is delegated to the `csv` crate so valid quoted CSV behaves like
//! Python's standard `csv` parser.
//!
//! The file reader requires the `csv` cargo feature (enabled by default).
//! [`decimal_count`] and [`detect_price_scale`] are pure string helpers and
//! stay available without it, so no-default-features consumers can still
//! derive a price scale for their own data.

#[cfg(feature = "csv")]
use std::error::Error;
#[cfg(feature = "csv")]
use std::fmt;
#[cfg(feature = "csv")]
use std::path::Path;

#[cfg(feature = "csv")]
use crate::types::Bar;

/// Errors returned while reading a TradingView CSV.
#[cfg(feature = "csv")]
#[derive(Debug)]
pub enum CsvError {
    /// The file could not be opened or read.
    Io(std::io::Error),
    /// The CSV structure itself could not be parsed.
    Csv(csv::Error),
    /// The file has no header row.
    Empty,
    /// A required `time/open/high/low/close` column is absent.
    MissingColumn(&'static str),
    /// A numeric cell failed to parse.
    ParseField {
        /// 1-based line number of the offending record.
        line: usize,
        /// Name of the column that failed to parse.
        column: &'static str,
        /// The raw cell content.
        value: String,
    },
}

#[cfg(feature = "csv")]
impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Csv(e) => write!(f, "csv error: {e}"),
            Self::Empty => write!(f, "csv is empty (no header row)"),
            Self::MissingColumn(c) => write!(f, "missing required column: {c}"),
            Self::ParseField {
                line,
                column,
                value,
            } => write!(
                f,
                "line {line}: cannot parse {column} value {value:?} as a number"
            ),
        }
    }
}

#[cfg(feature = "csv")]
impl Error for CsvError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Csv(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "csv")]
impl From<std::io::Error> for CsvError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[cfg(feature = "csv")]
impl From<csv::Error> for CsvError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value)
    }
}

/// Number of significant fractional digits in a decimal string
/// (`"0.91825"` -> 5, `"300"` -> 0, `"1.250"` -> 2).
#[must_use]
pub fn decimal_count(value: &str) -> usize {
    match value.split_once('.') {
        None => 0,
        Some((_, frac)) => frac.trim_end_matches('0').len(),
    }
}

/// Derives the ADX price scale (`10^max_decimals`, or `0.0` for integer data)
/// from the maximum fractional precision across all OHLC cells.
#[must_use]
pub fn detect_price_scale(rows: &[[String; 4]]) -> f64 {
    let mut max_decimals = 0;
    for row in rows {
        for cell in row {
            max_decimals = max_decimals.max(decimal_count(cell));
        }
    }
    if max_decimals == 0 {
        0.0
    } else {
        // Clamp the exponent to avoid i64 overflow on pathological input
        // (a cell with >=19 significant fractional digits would otherwise
        // panic in debug / wrap in release). 10^18 is the largest power of
        // ten that fits i64; real price data never exceeds a few decimals.
        10f64.powi(max_decimals.min(18) as i32)
    }
}

#[cfg(feature = "csv")]
pub(crate) fn column_index(
    header: &csv::StringRecord,
    name: &'static str,
) -> Result<usize, CsvError> {
    header
        .iter()
        .position(|c| c.trim().eq_ignore_ascii_case(name))
        .ok_or(CsvError::MissingColumn(name))
}

/// Reads OHLC bars and the derived price scale from a TradingView CSV file.
///
/// # Errors
/// Returns [`CsvError`] when the file cannot be read, is empty, lacks a
/// required `time/open/high/low/close` column, or contains an unparseable
/// numeric cell.
#[cfg(feature = "csv")]
pub fn read_tradingview_csv(path: &Path) -> Result<(Vec<Bar>, f64), CsvError> {
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

    let mut raw: Vec<(usize, String, [String; 4])> = Vec::new();
    for record in reader.records() {
        let record = record?;
        if record.iter().all(|cell| cell.trim().is_empty()) {
            continue;
        }
        let line_no = record
            .position()
            .map_or(0, |position| position.line() as usize);
        let cell = |i: usize| record.get(i).map_or("", str::trim).to_string();
        raw.push((
            line_no,
            cell(time_i),
            [cell(open_i), cell(high_i), cell(low_i), cell(close_i)],
        ));
    }

    let scales: Vec<[String; 4]> = raw.iter().map(|(_, _, ohlc)| ohlc.clone()).collect();
    let price_scale = detect_price_scale(&scales);

    let mut bars = Vec::with_capacity(raw.len());
    for (line_no, time, ohlc) in raw {
        let parse = |value: &str, column: &'static str| {
            value.parse::<f64>().map_err(|_| CsvError::ParseField {
                line: line_no,
                column,
                value: value.to_string(),
            })
        };
        bars.push(Bar {
            open: parse(&ohlc[0], "open")?,
            high: parse(&ohlc[1], "high")?,
            low: parse(&ohlc[2], "low")?,
            close: parse(&ohlc[3], "close")?,
            time,
        });
    }

    Ok((bars, price_scale))
}
