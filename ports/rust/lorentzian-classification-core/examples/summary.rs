//! Minimal end-to-end example: read a TradingView CSV, run the classifier with
//! default settings, and print a short summary of the last bar.
//!
//! Run with:
//! ```text
//! cargo run --example summary -p lorentzian-classification-core -- "your_export.csv"
//! ```
//!
//! Without an argument it falls back to the gold baseline vendored into the
//! crate, so `cargo run --example summary` works out of the box.

use std::path::Path;
use std::process::ExitCode;

use lorentzian_classification_core::{calculate, read_tradingview_csv, Settings};

fn main() -> ExitCode {
    let input = std::env::args().nth(1).unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/pine_btcusd_h1_trimmed_limited_history.csv")
            .to_string_lossy()
            .into_owned()
    });

    let (bars, price_scale) = match read_tradingview_csv(Path::new(&input)) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("error reading {input}: {err}");
            return ExitCode::FAILURE;
        }
    };

    let rows = calculate(&bars, &Settings::default(), price_scale);
    let Some(last) = rows.last() else {
        eprintln!("no rows produced (empty input?)");
        return ExitCode::FAILURE;
    };

    println!("bars:            {}", rows.len());
    println!("last time:       {}", last.bar.time);
    println!("last close:      {}", last.bar.close);
    println!("prediction:      {}", last.prediction);
    println!("direction:       {}", last.direction);
    println!("kernel estimate: {:.6}", last.kernel);
    println!(
        "trades: {} (wins {}, losses {})",
        last.total_trades, last.total_wins, last.total_losses
    );
    ExitCode::SUCCESS
}
