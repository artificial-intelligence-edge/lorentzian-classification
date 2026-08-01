//! Bit-faithful Rust port of the **Lorentzian Classification** indicator.
//!
//! This crate is a direct, forward-indexed port of the PineScript v6 reference
//! (and the parity-tested Python port under `ports/python`). Series use index
//! `0` as the oldest bar, and missing Pine `na` values are represented as
//! [`f64::NAN`] ([`MISSING`]).
//!
//! The implementation is deliberately allocation-simple and dependency-light so
//! it stays easy to audit against the reference. Floating-point operations
//! mirror the Python port's accumulation order and rounding (banker's rounding
//! via [`f64::round_ties_even`], libm `powf`/`exp`), which makes the two ports
//! bit-for-bit equal and keeps the crate within `1e-6` of the TradingView gold
//! baselines.
//!
//! # Cargo features
//!
//! * **`csv`** (default): TradingView CSV input ([`read_tradingview_csv`]) and
//!   Pine-export parity checking (the [`parity`] module), via the established
//!   `csv` crate so valid quoted CSV behaves like Python's standard `csv`
//!   parser. Disable it (`default-features = false`) for a zero-dependency
//!   numeric core when you feed [`Bar`] values from your own data pipeline.
//!
//! # Example
//! ```
//! use lorentzian_classification_core::{calculate, Bar, Settings};
//!
//! let bars: Vec<Bar> = (0..32)
//!     .map(|i| {
//!         let c = 100.0 + f64::from(i);
//!         Bar { time: i.to_string(), open: c, high: c + 1.0, low: c - 1.0, close: c }
//!     })
//!     .collect();
//! let rows = calculate(&bars, &Settings::default(), 0.0);
//! assert_eq!(rows.len(), bars.len());
//! ```

#![forbid(unsafe_code)]
// Published crate: every public item must be documented.
#![deny(missing_docs)]
#![warn(clippy::pedantic)]
// Numeric casts are pervasive in this index-heavy numeric port and are always
// within range for realistic bar counts; flagging each one adds noise.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
// The following pedantic lints conflict with this crate's stated goal of a
// 1:1, line-faithful translation of the PineScript / Python reference:
// `Settings`/`ResultRow` mirror the Pine inputs and outputs (many bools);
// paired bindings like `is_ema_up`/`is_sma_up` are intentionally parallel; the
// `x != 0.0 ? a : b` divide-by-zero guards mirror the reference expressions.
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::similar_names)]
#![allow(clippy::if_not_else)]
// doc_markdown flags every product name (PineScript, TradingView, …); not worth
// the backtick churn for prose.
#![allow(clippy::doc_markdown)]

pub mod ann;
pub mod csv_io;
pub mod display;
pub mod engine;
pub mod filters;
pub mod indicators;
pub mod kernels;
#[cfg(feature = "csv")]
pub mod parity;
pub mod types;

pub use ann::AnnState;
pub use csv_io::detect_price_scale;
#[cfg(feature = "csv")]
pub use csv_io::{read_tradingview_csv, CsvError};
pub use engine::{calculate, select_source};
#[cfg(feature = "csv")]
pub use parity::{parity_summary, read_pine_export, ExpectedRow, Mismatch, ParitySummary};
pub use types::{
    is_missing, nz, Bar, FeatureKind, FeatureSpec, ParseError, ResultRow, Settings, Source,
    MISSING, RESULT_FIELDNAMES,
};
