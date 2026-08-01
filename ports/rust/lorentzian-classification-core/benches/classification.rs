//! Criterion benchmark for the end-to-end `calculate` pipeline.
//!
//! Run with `cargo bench`. Uses the gold baseline vendored into the crate
//! (`tests/data/`), falling back to a synthetic series so the bench never
//! depends on external files.

use std::hint::black_box;
use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion};
use lorentzian_classification_core::{calculate, read_pine_export, Bar, Settings};

fn load_bars() -> Vec<Bar> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pine_btcusd_h1_trimmed_limited_history.csv");
    if let Ok((bars, _expected, _scale)) = read_pine_export(&path) {
        return bars;
    }
    // Synthetic fallback: a gently trending, oscillating series.
    (0..2_000)
        .map(|i| {
            let t = f64::from(i);
            let base = 100.0 + t * 0.05 + (t / 7.0).sin() * 5.0;
            Bar {
                time: i.to_string(),
                open: base,
                high: base + 1.5,
                low: base - 1.5,
                close: base + (t / 3.0).cos(),
            }
        })
        .collect()
}

fn bench_calculate(c: &mut Criterion) {
    let bars = load_bars();
    let settings = Settings::default();

    let mut group = c.benchmark_group("calculate");
    group.bench_function("default_limited_history", |b| {
        b.iter(|| calculate(black_box(&bars), black_box(&settings), 0.0));
    });
    group.bench_function("full_history", |b| {
        let full = Settings {
            include_full_history: true,
            ..Settings::default()
        };
        b.iter(|| calculate(black_box(&bars), black_box(&full), 0.0));
    });
    group.finish();
}

criterion_group!(benches, bench_calculate);
criterion_main!(benches);
