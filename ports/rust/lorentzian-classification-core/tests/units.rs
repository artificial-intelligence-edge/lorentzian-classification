//! Unit tests for the numeric helpers that are most at risk of porting drift:
//! banker's rounding in ADX quantization, decimal-scale detection, rescaling,
//! and the moving-average warmup semantics.

use lorentzian_classification_core::csv_io::{decimal_count, detect_price_scale};
use lorentzian_classification_core::indicators::{
    calc_rsi, ema, quant_sub, rescale, rma, sma, wilder_smooth,
};
use lorentzian_classification_core::{is_missing, RESULT_FIELDNAMES};

#[cfg(feature = "csv")]
fn temp_csv(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "lorentzian-rust-core-{}-{name}.csv",
        std::process::id()
    ))
}

#[test]
fn quant_sub_uses_bankers_rounding() {
    // scale 1.0: round-half-to-even, matching Python's round().
    // 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 3.5 -> 4.
    assert_eq!(quant_sub(2.5, 0.0, 1.0), 2.0);
    assert_eq!(quant_sub(3.5, 0.0, 1.0), 4.0);
    assert_eq!(quant_sub(0.5, 0.0, 1.0), 0.0);
    assert_eq!(quant_sub(1.5, 0.0, 1.0), 2.0);
}

#[test]
fn quant_sub_disabled_when_scale_non_positive() {
    let a = 1.234_567;
    let b = 0.001;
    assert_eq!(quant_sub(a, b, 0.0), a - b);
    assert_eq!(quant_sub(a, b, -1.0), a - b);
}

#[test]
fn quant_sub_quantizes_at_scale() {
    // With scale 100000 (5 decimals), values round at the 5th decimal.
    let scaled = quant_sub(1.234_565, 0.0, 100_000.0);
    // 1.234565 * 1e5 = 123456.5 -> round-half-even -> 123456 -> /1e5.
    assert!((scaled - 1.234_56).abs() < 1e-12, "got {scaled}");
}

#[test]
fn decimal_count_strips_trailing_zeros() {
    assert_eq!(decimal_count("0.91825"), 5);
    assert_eq!(decimal_count("300"), 0);
    assert_eq!(decimal_count("1.250"), 2);
    assert_eq!(decimal_count("1.0"), 0);
    assert_eq!(decimal_count("28963.4022646"), 7);
}

#[test]
fn price_scale_is_ten_to_max_decimals() {
    let rows = vec![
        ["0.9".into(), "0.91825".into(), "0.9".into(), "0.9".into()],
        ["1.0".into(), "1.0".into(), "1.0".into(), "1.0".into()],
    ];
    assert_eq!(detect_price_scale(&rows), 100_000.0);

    let integer_rows = vec![["300".into(), "370".into(), "300".into(), "370".into()]];
    assert_eq!(detect_price_scale(&integer_rows), 0.0);
}

#[test]
fn result_fieldnames_track_python_schema_surface() {
    assert_eq!(RESULT_FIELDNAMES.len(), 40);
    assert_eq!(
        &RESULT_FIELDNAMES[..5],
        ["time", "open", "high", "low", "close"]
    );
    assert_eq!(RESULT_FIELDNAMES[17], "Backtest Stream");
    assert_eq!(*RESULT_FIELDNAMES.last().unwrap(), "Win Rate");
}

#[cfg(feature = "csv")]
#[test]
fn tradingview_reader_accepts_quoted_csv_cells_like_python() {
    use lorentzian_classification_core::read_tradingview_csv;

    let path = temp_csv("quoted-input");
    std::fs::write(
        &path,
        "time,open,high,low,close\n\"2026-01-01, 00:00\",1.2345,1.2400,1.2300,1.2350\n",
    )
    .unwrap();

    let (bars, price_scale) = read_tradingview_csv(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(bars.len(), 1);
    assert_eq!(bars[0].time, "2026-01-01, 00:00");
    assert!((bars[0].open - 1.2345).abs() < 1e-12);
    assert_eq!(price_scale, 10_000.0);
}

#[test]
fn rescale_maps_ranges() {
    assert!((rescale(50.0, 0.0, 100.0, 0.0, 1.0) - 0.5).abs() < 1e-12);
    assert!((rescale(0.0, 0.0, 100.0, 0.0, 1.0)).abs() < 1e-12);
    assert!((rescale(100.0, 0.0, 100.0, 0.0, 1.0) - 1.0).abs() < 1e-12);
}

#[test]
fn sma_warmup_then_average() {
    let src = vec![1.0, 2.0, 3.0, 4.0];
    let out = sma(&src, 2);
    assert!(is_missing(out[0]));
    assert!((out[1] - 1.5).abs() < 1e-12);
    assert!((out[2] - 2.5).abs() < 1e-12);
    assert!((out[3] - 3.5).abs() < 1e-12);
}

#[test]
fn ema_seeds_with_sma_of_first_window() {
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let out = ema(&src, 3);
    assert!(is_missing(out[0]));
    assert!(is_missing(out[1]));
    // First defined value is the SMA of the first 3 elements.
    assert!((out[2] - 2.0).abs() < 1e-12);
    // Then EMA recursion with alpha = 2/(3+1) = 0.5.
    assert!((out[3] - (0.5 * 4.0 + 0.5 * 2.0)).abs() < 1e-12);
}

#[test]
fn rma_uses_wilder_alpha() {
    let src = vec![1.0, 2.0, 3.0, 4.0];
    let out = rma(&src, 2);
    assert!((out[1] - 1.5).abs() < 1e-12); // SMA seed of first 2
                                           // alpha = 1/2; out[2] = 0.5*3 + 0.5*1.5 = 2.25
    assert!((out[2] - 2.25).abs() < 1e-12);
}

#[test]
fn wilder_smooth_accumulates() {
    let src = vec![1.0, 1.0, 1.0];
    let out = wilder_smooth(&src, 2);
    assert!((out[0] - 1.0).abs() < 1e-12);
    // out[1] = 1 - 1/2 + 1 = 1.5; out[2] = 1.5 - 0.75 + 1 = 1.75
    assert!((out[1] - 1.5).abs() < 1e-12);
    assert!((out[2] - 1.75).abs() < 1e-12);
}

#[test]
fn rsi_is_bounded_and_full_on_monotonic_gains() {
    let src: Vec<f64> = (0..20).map(f64::from).collect();
    let out = calc_rsi(&src, 14);
    let last = *out.last().unwrap();
    assert!(!is_missing(last));
    // Monotonic increase -> zero losses -> RSI pegged at 100.
    assert!((last - 100.0).abs() < 1e-9, "got {last}");
}
