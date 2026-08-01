//! Lorentzian Classification under its canonical name.
//!
//! This crate is a thin alias that re-exports the entire public surface of
//! [`lorentzian-classification-core`](lorentzian_classification_core), so the
//! indicator can be depended on by its canonical name. Both spellings resolve
//! to the same implementation; depend on whichever reads better in your
//! `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! lorentzian-classification = "0.1"
//! ```
//!
//! The `csv` cargo feature (default) forwards to the core crate's `csv`
//! feature. See the core crate's documentation for the full API reference.
//!
//! # Example
//! ```
//! use lorentzian_classification::{calculate, Bar, Settings};
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
#![deny(missing_docs)]
#![warn(clippy::pedantic)]

pub use lorentzian_classification_core::*;
