//! Parity gate: the Rust port must reproduce the TradingView/Pine gold exports.
//!
//! These are the same baselines and the same contract (`tolerance = 1e-6` for
//! features/kernel, exact for prediction/direction/buy/sell/stops) used by the
//! Python parity suite. Because the Python port is already proven equal to these
//! Pine exports, passing here establishes Rust == Pine == Python transitively.
//!
//! One gold baseline is vendored into the crate (`tests/data/`), so the
//! published package remains parity-testable outside the monorepo. The larger
//! baselines live at the repository root and their tests skip gracefully when
//! that root is absent (i.e. when running from an extracted crates.io package).

use std::path::{Path, PathBuf};

use lorentzian_classification_core::{calculate, parity_summary, read_pine_export, Settings};

/// The gold baseline vendored into the crate so the packaged crate stays
/// self-contained.
const VENDORED_BASELINE: &str = "pine_btcusd_h1_trimmed_limited_history.csv";

fn vendored_baseline() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(VENDORED_BASELINE)
}

/// Full monorepo baseline path (`<repo>/tests/parity/baselines/<filename>`),
/// or `None` when running outside the monorepo (extracted package).
fn repo_baseline(filename: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)?
        .join("tests/parity/baselines")
        .join(filename);
    path.exists().then_some(path)
}

fn temp_csv(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "lorentzian-rust-parity-{}-{name}.csv",
        std::process::id()
    ))
}

/// Runs the parity comparison for one baseline and asserts an exact pass.
fn assert_baseline_parity(path: &Path, include_full_history: bool) {
    let filename = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (bars, expected, price_scale) = read_pine_export(path).expect("baseline CSV should parse");
    let settings = Settings {
        include_full_history,
        ..Settings::default()
    };
    let results = calculate(&bars, &settings, price_scale);

    assert_eq!(
        results.len(),
        expected.len(),
        "{filename}: row count mismatch"
    );

    let summary = parity_summary(&expected, &results, 1e-6, &settings);
    assert!(
        summary.pass,
        "{filename}: parity FAILED — compared {} rows from index {}, \
         max feature diff {:e}, max kernel diff {:e}; first mismatches: {:?}",
        summary.compared,
        summary.max_bars_back_index,
        summary.max_feature_diff,
        summary.max_kernel_diff,
        &summary.mismatches[..summary.mismatches.len().min(5)],
    );
    assert!(summary.compared > 0, "{filename}: nothing was compared");
}

/// Runs a monorepo-only baseline, skipping when the repo root is absent.
fn assert_repo_baseline_parity(filename: &str, include_full_history: bool) {
    let Some(path) = repo_baseline(filename) else {
        eprintln!("skipping {filename}: monorepo gold baseline not present in the packaged crate");
        return;
    };
    assert_baseline_parity(&path, include_full_history);
}

#[test]
fn oanda_eurusd_daily_full_history() {
    assert_repo_baseline_parity("pine_oanda_eurusd_1d_full_history.csv", true);
}

#[test]
fn tastyfx_eurusd_daily_full_history() {
    assert_repo_baseline_parity("pine_tastyfx_eurusd_1d_full_history.csv", true);
}

#[test]
fn coinbase_btcusd_daily_limited_history() {
    assert_repo_baseline_parity("pine_coinbase_btcusd_1d_limited_history.csv", false);
}

#[test]
fn btcusd_h1_trimmed_limited_history() {
    // Vendored fixture: always present, including in the extracted package.
    assert_baseline_parity(&vendored_baseline(), false);
}

/// The vendored fixture must stay byte-identical to the monorepo gold
/// baseline it was copied from (skipped in the extracted package, where the
/// monorepo copy is not available).
#[test]
fn vendored_fixture_matches_monorepo_baseline() {
    let Some(repo) = repo_baseline(VENDORED_BASELINE) else {
        eprintln!("skipping vendored-fixture drift check: monorepo baseline not present");
        return;
    };
    let repo_bytes = std::fs::read(repo).expect("read monorepo baseline");
    let vendored_bytes = std::fs::read(vendored_baseline()).expect("read vendored fixture");
    assert!(
        repo_bytes == vendored_bytes,
        "vendored tests/data/{VENDORED_BASELINE} has drifted from \
         tests/parity/baselines/{VENDORED_BASELINE}; re-copy the gold baseline"
    );
}

/// Determinism: recomputing the same input yields identical results.
///
/// Result rows carry `NaN` trade-stat ratios during warmup, and `NaN != NaN`,
/// so the rows are compared via their `Debug` rendering (where `NaN` formats
/// identically) rather than `PartialEq`.
#[test]
fn recompute_is_deterministic() {
    let (bars, _expected, price_scale) =
        read_pine_export(&vendored_baseline()).expect("parse vendored baseline");
    let settings = Settings::default();
    let a = calculate(&bars, &settings, price_scale);
    let b = calculate(&bars, &settings, price_scale);
    assert_eq!(
        format!("{a:?}"),
        format!("{b:?}"),
        "calculate must be deterministic"
    );
}

#[test]
fn pine_export_reader_accepts_quoted_csv_cells_like_python() {
    let path = temp_csv("quoted-pine-export");
    std::fs::write(
        &path,
        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell\n\"2026-01-01, 00:00\",1.2345,1.2400,1.2300,1.2350,0.1,0.2,0.3,0.4,0.5,1.1,-1,-1,0,1,,1\n",
    )
    .unwrap();

    let (bars, expected, price_scale) = read_pine_export(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(bars.len(), 1);
    assert_eq!(bars[0].time, "2026-01-01, 00:00");
    assert_eq!(expected.len(), 1);
    assert_eq!(expected[0].features, [0.1, 0.2, 0.3, 0.4, 0.5]);
    assert_eq!(expected[0].prediction, -1);
    assert!(expected[0].sell);
    assert!(expected[0].stop_sell);
    assert_eq!(price_scale, 10_000.0);
}
