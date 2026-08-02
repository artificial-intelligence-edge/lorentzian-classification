"""Lorentzian Classification business logic.

This module is a direct, forward-indexed port of the PineScript v6 original
and the parity-tested external implementation. Arrays use index 0 as the oldest
bar. Missing Pine ``na`` values are represented as ``math.nan``.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .settings import Settings


MISSING = math.nan


@dataclass(frozen=True)
class Bar:
    time: str
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class TvRow:
    bar: Bar
    f1: float = MISSING
    f2: float = MISSING
    f3: float = MISSING
    f4: float = MISSING
    f5: float = MISSING
    kernel: float = MISSING
    prediction: int = 0
    direction: int = 0
    buy: bool = False
    sell: bool = False
    exit_buy: bool = False
    exit_sell: bool = False
    backtest_stream: int | None = None
    open_long_alert: bool | None = None
    close_long_alert: bool | None = None
    open_short_alert: bool | None = None
    close_short_alert: bool | None = None
    open_position_alert: bool | None = None
    close_position_alert: bool | None = None
    kernel_bullish_alert: bool | None = None
    kernel_bearish_alert: bool | None = None
    kernel_plot_color: str | None = None
    prediction_label: str | None = None
    prediction_label_y: float | None = None
    prediction_label_color: str | None = None
    bar_color: str | None = None
    trade_stats_visible: bool | None = None
    trade_stats_header: str | None = None
    total_wins: int | None = None
    total_losses: int | None = None
    total_early_signal_flips: int | None = None
    total_trades: int | None = None
    win_loss_ratio: float | None = None
    table_wl_ratio: float | None = None
    win_rate: float | None = None


@dataclass(frozen=True)
class ResultRow:
    bar: Bar
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    kernel: float
    prediction: int
    direction: int
    buy: bool
    sell: bool
    exit_buy: bool
    exit_sell: bool
    stop_buy: bool
    stop_sell: bool
    backtest_stream: int
    open_long_alert: bool
    close_long_alert: bool
    open_short_alert: bool
    close_short_alert: bool
    open_position_alert: bool
    close_position_alert: bool
    kernel_bullish_alert: bool
    kernel_bearish_alert: bool
    kernel_plot_color: str
    prediction_label: str
    prediction_label_y: float
    prediction_label_color: str
    bar_color: str
    trade_stats_header: str
    total_wins: int
    total_losses: int
    total_early_signal_flips: int
    total_trades: int
    win_loss_ratio: float
    table_wl_ratio: float
    win_rate: float
    trade_stats_visible: bool = True


RESULT_FIELDNAMES = [
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
]

OHLC_COLUMNS = ("open", "high", "low", "close")
TIME_COLUMNS = ("time", "date")


def is_missing(value: float) -> bool:
    return math.isnan(value)


def nz(value: float, fallback: float = 0.0) -> float:
    return fallback if is_missing(value) else value


def parse_float(value: str) -> float:
    return MISSING if value == "" else float(value)


def parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "na", "nan"}


def parse_optional_bool(row: dict[str, str], key: str) -> bool | None:
    if key not in row:
        return None
    return parse_bool(row.get(key))


def parse_optional_int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key)
    if value in ("", None):
        return None
    return int(float(value))


def parse_optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in ("", None):
        return None
    return float(value)


def parse_optional_string(row: dict[str, str], key: str) -> str | None:
    if key not in row:
        return None
    value = row.get(key)
    return "" if value is None else value


def parse_optional_prediction_label(row: dict[str, str]) -> str | None:
    value = parse_optional_string(row, "Prediction Label")
    if value in (None, ""):
        return value
    try:
        numeric = float(value)
    except ValueError:
        return value
    return str(int(numeric)) if numeric.is_integer() else value


def decode_optional_color_code(value: str | None) -> str | None:
    if value in ("", None):
        return ""
    try:
        code = int(round(float(value)))
    except ValueError:
        return value
    if code == 0:
        return pine_color("#000000", 100)
    if 100 <= code < 200:
        return pine_color("#009988", code - 100)
    if -200 < code <= -100:
        return pine_color("#CC3311", abs(code) - 100)
    if 200 <= code < 300:
        return pine_color("#787b86", code - 200)
    return value


def parse_optional_color_string(row: dict[str, str], key: str) -> str | None:
    if key not in row:
        return None
    return decode_optional_color_code(row.get(key))


def parse_optional_trade_stats_header(row: dict[str, str]) -> str | None:
    value = parse_optional_string(row, "Trade Stats Header")
    if value in (None, ""):
        return value
    try:
        code = int(round(float(value)))
    except ValueError:
        return value
    return "\U0001f4c8 Trade Stats" if code == 1 else value


def get_feature_slot(row: dict[str, str], slot: str, default_column: str) -> str:
    if default_column in row:
        return row.get(default_column, "")
    prefix = f"{slot}_"
    for key, value in row.items():
        if key.upper().startswith(prefix):
            return value
    return ""


def decimal_count(value: str) -> int:
    if "." not in value:
        return 0
    decimals = value.split(".", 1)[1].rstrip("0")
    return len(decimals)


def detect_price_scale(rows: list[dict[str, str]]) -> float:
    max_decimals = 0
    for row in rows:
        for key in ("open", "high", "low", "close"):
            max_decimals = max(max_decimals, decimal_count(row.get(key, "")))
    # Clamp the exponent to mirror the Rust port (which must avoid i64
    # overflow on >=19-digit cells); keeps the two ports bit-identical.
    return float(10 ** min(max_decimals, 18)) if max_decimals else 0.0


def normalize_record(record: Mapping[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in record.items():
        if value is None:
            normalized[str(key)] = ""
        elif isinstance(value, float) and math.isnan(value):
            normalized[str(key)] = ""
        else:
            normalized[str(key)] = str(value)
    return normalized


def lookup_column(row: Mapping[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in row:
            return candidate
    casefolded = {key.casefold(): key for key in row}
    for candidate in candidates:
        key = casefolded.get(candidate.casefold())
        if key is not None:
            return key
    return None


def ensure_ohlc_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(records):
        canonical = dict(row)
        time_key = lookup_column(canonical, TIME_COLUMNS)
        if time_key is None:
            raise ValueError(f"record {index}: missing time/date column")
        canonical["time"] = canonical[time_key]
        for column in OHLC_COLUMNS:
            source_key = lookup_column(canonical, [column])
            if source_key is None:
                raise ValueError(f"record {index}: missing {column} column")
            canonical[column] = canonical[source_key]
        normalized.append(canonical)
    return normalized


def dataframe_records(data: object) -> list[dict[str, str]] | None:
    to_dict = getattr(data, "to_dict", None)
    columns = getattr(data, "columns", None)
    if not callable(to_dict) or columns is None:
        return None

    raw_records = to_dict("records")
    if not isinstance(raw_records, list):
        raise TypeError("dataframe-like input must return a list from to_dict('records')")

    records = [normalize_record(record) for record in raw_records if isinstance(record, Mapping)]
    column_names = {str(column).casefold() for column in columns}
    if not any(column in column_names for column in TIME_COLUMNS):
        index = list(getattr(data, "index", range(len(records))))
        if len(index) != len(records):
            raise ValueError("dataframe-like input index length does not match rows")
        for row, index_value in zip(records, index, strict=True):
            row["time"] = str(index_value)
    return records


def rows_from_records(
    records: Iterable[Mapping[str, object]],
    feature_columns: dict[str, str] | None = None,
) -> tuple[list[TvRow], float]:
    rows = ensure_ohlc_records([normalize_record(record) for record in records])
    return parse_tradingview_rows(rows, feature_columns=feature_columns)


def coerce_input_rows(
    data: str | Path | Iterable[Bar] | Iterable[TvRow] | Iterable[Mapping[str, object]] | object,
    feature_columns: dict[str, str] | None = None,
) -> tuple[list[TvRow], float]:
    if isinstance(data, str | Path):
        return read_tradingview_csv(data, feature_columns=feature_columns)

    records = dataframe_records(data)
    if records is not None:
        return rows_from_records(records, feature_columns=feature_columns)

    materialized = list(data)  # type: ignore[arg-type]
    if not materialized:
        return [], 0.0
    first = materialized[0]
    if isinstance(first, TvRow):
        return list(materialized), 0.0  # type: ignore[list-item]
    if isinstance(first, Bar):
        return [TvRow(bar=row) for row in materialized], 0.0  # type: ignore[arg-type]
    if isinstance(first, Mapping):
        return rows_from_records(materialized, feature_columns=feature_columns)  # type: ignore[arg-type]
    raise TypeError("data must be a CSV path, Bar/TvRow iterable, mapping records, or dataframe-like object")


def parse_tradingview_rows(
    rows: list[dict[str, str]],
    feature_columns: dict[str, str] | None = None,
) -> tuple[list[TvRow], float]:
    price_scale = detect_price_scale(rows)
    feature_columns = feature_columns or {}
    parsed: list[TvRow] = []
    for row in rows:
        bar = Bar(
            time=row["time"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
        )
        parsed.append(
            TvRow(
                bar=bar,
                f1=parse_float(get_feature_slot(row, "F1", feature_columns.get("F1", "F1_RSI"))),
                f2=parse_float(get_feature_slot(row, "F2", feature_columns.get("F2", "F2_WT"))),
                f3=parse_float(get_feature_slot(row, "F3", feature_columns.get("F3", "F3_CCI"))),
                f4=parse_float(get_feature_slot(row, "F4", feature_columns.get("F4", "F4_ADX"))),
                f5=parse_float(get_feature_slot(row, "F5", feature_columns.get("F5", "F5_RSI9"))),
                kernel=parse_float(row.get("Kernel Regression Estimate", "")),
                prediction=int(float(row.get("Prediction") or 0)),
                direction=int(float(row.get("Direction") or 0)),
                buy=parse_bool(row.get("Buy", "")),
                sell=parse_bool(row.get("Sell", "")),
                exit_buy=parse_bool(row.get("StopBuy", "")),
                exit_sell=parse_bool(row.get("StopSell", "")),
                backtest_stream=(
                    int(float(row["Backtest Stream"]))
                    if row.get("Backtest Stream", "") not in ("", None)
                    else None
                ),
                open_long_alert=parse_optional_bool(row, "Open Long Alert"),
                close_long_alert=parse_optional_bool(row, "Close Long Alert"),
                open_short_alert=parse_optional_bool(row, "Open Short Alert"),
                close_short_alert=parse_optional_bool(row, "Close Short Alert"),
                open_position_alert=parse_optional_bool(row, "Open Position Alert"),
                close_position_alert=parse_optional_bool(row, "Close Position Alert"),
                kernel_bullish_alert=parse_optional_bool(row, "Kernel Bullish Alert"),
                kernel_bearish_alert=parse_optional_bool(row, "Kernel Bearish Alert"),
                kernel_plot_color=parse_optional_color_string(row, "Kernel Plot Color"),
                prediction_label=parse_optional_prediction_label(row),
                prediction_label_y=parse_optional_float(row, "Prediction Label Y"),
                prediction_label_color=parse_optional_color_string(row, "Prediction Label Color"),
                bar_color=parse_optional_color_string(row, "Bar Color"),
                trade_stats_visible=parse_optional_bool(row, "Trade Stats Visible"),
                trade_stats_header=parse_optional_trade_stats_header(row),
                total_wins=parse_optional_int(row, "Total Wins"),
                total_losses=parse_optional_int(row, "Total Losses"),
                total_early_signal_flips=parse_optional_int(row, "Total Early Signal Flips"),
                total_trades=parse_optional_int(row, "Total Trades"),
                win_loss_ratio=parse_optional_float(row, "Win Loss Ratio"),
                table_wl_ratio=parse_optional_float(row, "Table WL Ratio"),
                win_rate=parse_optional_float(row, "Win Rate"),
            )
        )
    return parsed, price_scale


def read_tradingview_csv(path: str | Path, feature_columns: dict[str, str] | None = None) -> tuple[list[TvRow], float]:
    with Path(path).open(newline="") as handle:
        rows = ensure_ohlc_records(list(csv.DictReader(handle)))
    return parse_tradingview_rows(rows, feature_columns=feature_columns)


def format_result_float(value: float, digits: int = 16) -> str:
    return "" if is_missing(value) else f"{value:.{digits}f}"


def sanitize_text(value: str) -> str:
    """Neutralize spreadsheet formula injection (CWE-1236) in cells copied
    verbatim from input. A leading ``= + - @`` or control character is what a
    spreadsheet treats as a formula; prefixing with ``'`` makes it inert. Real
    timestamps never begin with these, so this is a no-op for legitimate data
    and keeps the output bit-identical to the Rust port."""
    if value[:1] in ("=", "+", "-", "@", "\t", "\r", "\n"):
        return "'" + value
    return value


def result_to_csv_row(row: ResultRow) -> list[object]:
    return [
        sanitize_text(str(row.bar.time)),
        f"{row.bar.open:.8f}",
        f"{row.bar.high:.8f}",
        f"{row.bar.low:.8f}",
        f"{row.bar.close:.8f}",
        format_result_float(row.f1),
        format_result_float(row.f2),
        format_result_float(row.f3),
        format_result_float(row.f4),
        format_result_float(row.f5),
        format_result_float(row.kernel),
        row.prediction,
        row.direction,
        "1" if row.buy else "",
        "1" if row.sell else "",
        "1" if row.stop_buy else "",
        "1" if row.stop_sell else "",
        row.backtest_stream if row.backtest_stream else "",
        "1" if row.open_long_alert else "",
        "1" if row.close_long_alert else "",
        "1" if row.open_short_alert else "",
        "1" if row.close_short_alert else "",
        "1" if row.open_position_alert else "",
        "1" if row.close_position_alert else "",
        "1" if row.kernel_bullish_alert else "",
        "1" if row.kernel_bearish_alert else "",
        row.kernel_plot_color,
        row.prediction_label,
        format_result_float(row.prediction_label_y),
        row.prediction_label_color,
        row.bar_color,
        "1" if row.trade_stats_visible else "",
        row.trade_stats_header,
        row.total_wins,
        row.total_losses,
        row.total_early_signal_flips,
        row.total_trades,
        format_result_float(row.win_loss_ratio),
        format_result_float(row.table_wl_ratio),
        format_result_float(row.win_rate),
    ]


def result_to_mapping(row: ResultRow) -> dict[str, object]:
    return dict(zip(RESULT_FIELDNAMES, result_to_csv_row(row), strict=True))


def write_result_csv(path: str | Path, rows: Iterable[ResultRow]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_FIELDNAMES)
        writer.writerows(result_to_csv_row(row) for row in rows)


class LorentzianClassification:
    """Library-facing wrapper around the parity-tested calculation path."""

    def __init__(
        self,
        data: str | Path | Iterable[Bar] | Iterable[TvRow] | Iterable[Mapping[str, object]] | object,
        settings: Settings | None = None,
        *,
        price_scale: float | None = None,
        feature_columns: dict[str, str] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.rows, detected_price_scale = coerce_input_rows(data, feature_columns=feature_columns)
        self.price_scale = detected_price_scale if price_scale is None else price_scale
        self.results = calculate(self.rows, self.settings, price_scale=self.price_scale)

    @property
    def latest(self) -> ResultRow:
        if not self.results:
            raise ValueError("no Lorentzian Classification results available")
        return self.results[-1]

    @property
    def data(self) -> list[ResultRow]:
        return self.results

    def to_records(self) -> list[dict[str, object]]:
        return [result_to_mapping(row) for row in self.results]

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("install lorentzian-classification[dataframe] to use to_dataframe()") from exc
        return pd.DataFrame(self.to_records(), columns=RESULT_FIELDNAMES)

    def dump(self, path: str | Path) -> Path:
        output_path = Path(path)
        write_result_csv(output_path, self.results)
        return output_path

    def to_csv(self, path: str | Path) -> Path:
        return self.dump(path)


def select_source(bars: list[Bar], source: str) -> list[float]:
    source = source.lower()
    if source == "open":
        return [bar.open for bar in bars]
    if source == "high":
        return [bar.high for bar in bars]
    if source == "low":
        return [bar.low for bar in bars]
    if source == "close":
        return [bar.close for bar in bars]
    if source == "hl2":
        return [(bar.high + bar.low) / 2.0 for bar in bars]
    if source == "hlc3":
        return [(bar.high + bar.low + bar.close) / 3.0 for bar in bars]
    if source == "ohlc4":
        return [(bar.open + bar.high + bar.low + bar.close) / 4.0 for bar in bars]
    raise ValueError(f"unsupported source: {source}")


def pine_color(hex_color: str, transparency: int = 0) -> str:
    return f"{hex_color}@{transparency}"


def color_with_transparency(color: str, transparency: int) -> str:
    base = color.split("@", 1)[0]
    return pine_color(base, transparency)


def prediction_green(prediction: float, use_confidence_gradient: bool) -> str:
    if not use_confidence_gradient:
        return pine_color("#009988", 0)
    scaled = min(abs(prediction), 10)
    if scaled >= 9:
        return pine_color("#009988", 0)
    if scaled >= 8:
        return pine_color("#009988", 10)
    if scaled >= 7:
        return pine_color("#009988", 20)
    if scaled >= 6:
        return pine_color("#009988", 30)
    if scaled >= 5:
        return pine_color("#009988", 40)
    if scaled >= 4:
        return pine_color("#009988", 50)
    if scaled >= 3:
        return pine_color("#009988", 60)
    if scaled >= 2:
        return pine_color("#009988", 70)
    if scaled >= 1:
        return pine_color("#009988", 80)
    return pine_color("#009988", 90)


def prediction_red(prediction: float, use_confidence_gradient: bool) -> str:
    if not use_confidence_gradient:
        return pine_color("#CC3311", 0)
    scaled = min(abs(prediction), 10)
    if scaled >= 9:
        return pine_color("#CC3311", 0)
    if scaled >= 8:
        return pine_color("#CC3311", 10)
    if scaled >= 7:
        return pine_color("#CC3311", 20)
    if scaled >= 6:
        return pine_color("#CC3311", 30)
    if scaled >= 5:
        return pine_color("#CC3311", 40)
    if scaled >= 4:
        return pine_color("#CC3311", 50)
    if scaled >= 3:
        return pine_color("#CC3311", 60)
    if scaled >= 2:
        return pine_color("#CC3311", 70)
    if scaled >= 1:
        return pine_color("#CC3311", 80)
    return pine_color("#CC3311", 90)


def sma(src: list[float], period: int) -> list[float]:
    out = [MISSING] * len(src)
    if period <= 0:
        return out
    for i in range(len(src)):
        if i < period - 1:
            continue
        window = src[i - period + 1 : i + 1]
        if any(is_missing(v) for v in window):
            continue
        out[i] = sum(window) / period
    return out


def ema(src: list[float], period: int) -> list[float]:
    out = [MISSING] * len(src)
    if period <= 0:
        return out
    alpha = 2.0 / (period + 1)
    for i, value in enumerate(src):
        if is_missing(value):
            continue
        if i > 0 and not is_missing(out[i - 1]):
            out[i] = alpha * value + (1.0 - alpha) * out[i - 1]
            continue
        if i >= period - 1:
            window = src[i - period + 1 : i + 1]
            if not any(is_missing(v) for v in window):
                out[i] = sum(window) / period
    return out


def rma(src: list[float], period: int) -> list[float]:
    out = [MISSING] * len(src)
    if period <= 0:
        return out
    alpha = 1.0 / period
    for i, value in enumerate(src):
        if is_missing(value):
            continue
        if i > 0 and not is_missing(out[i - 1]):
            out[i] = alpha * value + (1.0 - alpha) * out[i - 1]
            continue
        if i >= period - 1:
            window = src[i - period + 1 : i + 1]
            if not any(is_missing(v) for v in window):
                out[i] = sum(window) / period
    return out


def wilder_smooth(src: list[float], period: int) -> list[float]:
    out = [MISSING] * len(src)
    if period <= 0:
        return out
    for i, value in enumerate(src):
        if i == 0 or is_missing(out[i - 1]):
            out[i] = value
        else:
            out[i] = out[i - 1] - out[i - 1] / period + value
    return out


def rescale(src: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    return new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 1e-9)


def normalize_running(src: list[float], out_min: float = 0.0, out_max: float = 1.0) -> list[float]:
    out = [MISSING] * len(src)
    historic_min = 1e11
    historic_max = -1e11
    for i, value in enumerate(src):
        if is_missing(value):
            continue
        historic_min = min(historic_min, value)
        historic_max = max(historic_max, value)
        out[i] = out_min + (out_max - out_min) * (value - historic_min) / max(
            historic_max - historic_min, 1e-9
        )
    return out


def calc_rsi(src: list[float], period: int) -> list[float]:
    gain = [MISSING] * len(src)
    loss = [MISSING] * len(src)
    for i in range(1, len(src)):
        change = src[i] - src[i - 1]
        gain[i] = change if change > 0 else 0.0
        loss[i] = -change if change < 0 else 0.0
    gain_rma = rma(gain, period)
    loss_rma = rma(loss, period)
    out = [MISSING] * len(src)
    for i in range(len(src)):
        if is_missing(gain_rma[i]) or is_missing(loss_rma[i]):
            continue
        if loss_rma[i] == 0:
            out[i] = 100.0
        else:
            rs = gain_rma[i] / loss_rma[i]
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def calc_normalized_rsi(src: list[float], n1: int, n2: int) -> list[float]:
    smoothed = ema(calc_rsi(src, n1), n2)
    return [rescale(v, 0, 100, 0, 1) if not is_missing(v) else MISSING for v in smoothed]


def calc_cci(src: list[float], period: int) -> list[float]:
    avg = sma(src, period)
    out = [MISSING] * len(src)
    for i in range(len(src)):
        if is_missing(avg[i]):
            continue
        window = src[i - period + 1 : i + 1]
        mean_dev = sum(abs(v - avg[i]) for v in window) / period
        out[i] = (src[i] - avg[i]) / (0.015 * mean_dev) if mean_dev != 0 else 0.0
    return out


def calc_normalized_cci(src: list[float], n1: int, n2: int) -> list[float]:
    return normalize_running(ema(calc_cci(src, n1), n2))


def calc_wavetrend(hlc3: list[float], n1: int, n2: int) -> list[float]:
    ema1 = ema(hlc3, n1)
    abs_dev = [
        abs(hlc3[i] - ema1[i]) if not is_missing(ema1[i]) else MISSING
        for i in range(len(hlc3))
    ]
    ema2 = ema(abs_dev, n1)
    ci = [MISSING] * len(hlc3)
    for i in range(len(hlc3)):
        if is_missing(ema1[i]) or is_missing(ema2[i]):
            continue
        ci[i] = (hlc3[i] - ema1[i]) / (0.015 * ema2[i]) if ema2[i] != 0 else 0.0
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, 4)
    raw = [
        wt1[i] - wt2[i] if not is_missing(wt1[i]) and not is_missing(wt2[i]) else MISSING
        for i in range(len(hlc3))
    ]
    return normalize_running(raw)


def quant_sub(a: float, b: float, scale: float) -> float:
    if scale <= 0:
        return a - b
    return (round(a * scale) - round(b * scale)) / scale


def calc_atr(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    tr = []
    for i in range(len(close)):
        prev_close = close[i - 1] if i > 0 else 0.0
        tr.append(max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close)))
    return rma(tr, period)


def calc_adx(
    high: list[float], low: list[float], close: list[float], period: int, price_scale: float
) -> list[float]:
    tr: list[float] = []
    dm_plus: list[float] = []
    dm_minus: list[float] = []
    for i in range(len(close)):
        prev_close = close[i - 1] if i > 0 else 0.0
        prev_high = high[i - 1] if i > 0 else 0.0
        prev_low = low[i - 1] if i > 0 else 0.0
        tr.append(
            max(
                quant_sub(high[i], low[i], price_scale),
                abs(quant_sub(high[i], prev_close, price_scale)),
                abs(quant_sub(low[i], prev_close, price_scale)),
            )
        )
        up_move = quant_sub(high[i], prev_high, price_scale)
        down_move = quant_sub(prev_low, low[i], price_scale)
        dm_plus.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        dm_minus.append(down_move if down_move > up_move and down_move > 0 else 0.0)
    tr_smooth = wilder_smooth(tr, period)
    plus_smooth = wilder_smooth(dm_plus, period)
    minus_smooth = wilder_smooth(dm_minus, period)
    dx = []
    for i in range(len(close)):
        di_plus = plus_smooth[i] / tr_smooth[i] * 100 if tr_smooth[i] != 0 else 0.0
        di_minus = minus_smooth[i] / tr_smooth[i] * 100 if tr_smooth[i] != 0 else 0.0
        dx.append(abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if di_plus + di_minus else 0.0)
    adx_rma = rma(dx, period)
    return [rescale(v, 0, 100, 0, 1) if not is_missing(v) else MISSING for v in adx_rma]


def calc_raw_adx_filter(high: list[float], low: list[float], close: list[float], period: int) -> list[float]:
    normalized = calc_adx(high, low, close, period, 0.0)
    return [v * 100.0 if not is_missing(v) else 0.0 for v in normalized]


def calc_feature(
    spec: tuple[str, int, int],
    close: list[float],
    high: list[float],
    low: list[float],
    hlc3: list[float],
    price_scale: float,
) -> list[float]:
    kind, a, b = spec
    if kind == "RSI":
        return calc_normalized_rsi(close, a, b)
    if kind == "WT":
        return calc_wavetrend(hlc3, a, b)
    if kind == "CCI":
        return calc_normalized_cci(close, a, b)
    if kind == "ADX":
        return calc_adx(high, low, close, a, price_scale)
    raise ValueError(f"unsupported feature type: {kind}")


def calc_regime_filter(ohlc4: list[float], high: list[float], low: list[float]) -> tuple[list[float], list[float]]:
    n = len(ohlc4)
    abs_slope = [0.0] * n
    ema_abs = [0.0] * n
    if not n:
        return abs_slope, ema_abs
    value1 = [0.0] * n
    value2 = [0.0] * n
    klmf = [0.0] * n
    value2[0] = high[0] - low[0]
    klmf[0] = ohlc4[0]
    alpha_ema = 2.0 / 201.0
    for i in range(1, n):
        value1[i] = 0.2 * (ohlc4[i] - ohlc4[i - 1]) + 0.8 * nz(value1[i - 1])
        value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * nz(value2[i - 1])
        omega = abs(value1[i] / value2[i]) if value2[i] != 0 else 0.0
        alpha = (-(omega**2) + math.sqrt(omega**4 + 16.0 * omega**2)) / 8.0
        klmf[i] = alpha * ohlc4[i] + (1.0 - alpha) * nz(klmf[i - 1])
        abs_slope[i] = abs(klmf[i] - klmf[i - 1])
        prev_ema = nz(ema_abs[i - 1])
        if prev_ema == 0 and i < 200:
            ema_abs[i] = abs_slope[i]
        else:
            ema_abs[i] = alpha_ema * abs_slope[i] + (1.0 - alpha_ema) * prev_ema
    return abs_slope, ema_abs


def kernel_rational_quadratic(
    src: list[float], bar_index: int, lookback: int, relative_weight: float, start_at_bar: int
) -> float:
    current_weight = 0.0
    cumulative_weight = 0.0
    denom = max(float(lookback**2) * 2.0 * relative_weight, 1e-10)
    for i in range(min(1 + start_at_bar, bar_index) + 1):
        weight = (1.0 + (i**2 / denom)) ** (-relative_weight)
        current_weight += src[bar_index - i] * weight
        cumulative_weight += weight
    return current_weight / cumulative_weight if cumulative_weight > 0 else src[bar_index]


def kernel_gaussian(src: list[float], bar_index: int, lookback: int, start_at_bar: int) -> float:
    current_weight = 0.0
    cumulative_weight = 0.0
    denom = max(2.0 * float(lookback**2), 1e-10)
    for i in range(min(1 + start_at_bar, bar_index) + 1):
        weight = math.exp(-(i**2) / denom)
        current_weight += src[bar_index - i] * weight
        cumulative_weight += weight
    return current_weight / cumulative_weight if cumulative_weight > 0 else src[bar_index]


class AnnState:
    def __init__(self) -> None:
        self.features: list[list[float]] = [[], [], [], [], []]
        self.labels: list[int] = []
        self.distances: list[float] = []
        self.predictions: list[int] = []

    def push(self, values: tuple[float, float, float, float, float], label: int) -> None:
        for idx, value in enumerate(values):
            self.features[idx].append(value)
        self.labels.append(label)

    def distance(self, idx: int, values: tuple[float, float, float, float, float], feature_count: int) -> float:
        distance = 0.0
        for fidx in range(feature_count):
            current = values[fidx]
            historical = self.features[fidx][idx]
            if is_missing(current) or is_missing(historical):
                return -math.inf
            distance += math.log(1.0 + abs(current - historical))
        return distance

    def run(
        self,
        values: tuple[float, float, float, float, float],
        settings: Settings,
        last_bar_index: int,
    ) -> int:
        if not self.labels:
            return 0
        size_loop = min(settings.max_bars_back - 1, len(self.labels) - 1)
        max_bars_back_index = last_bar_index - settings.max_bars_back if last_bar_index >= settings.max_bars_back else 0
        start_idx = 0 if settings.include_full_history else max_bars_back_index
        last_distance = -1.0
        if start_idx <= size_loop:
            iterable = range(start_idx, size_loop + 1)
        else:
            iterable = range(start_idx, size_loop - 1, -1)
        for idx in iterable:
            distance = self.distance(idx, values, settings.feature_count)
            if distance >= last_distance and idx % 4 != 0:
                last_distance = distance
                self.distances.append(distance)
                self.predictions.append(round(self.labels[idx]))
                if len(self.predictions) > settings.neighbors_count:
                    threshold_idx = round(settings.neighbors_count * 3.0 / 4.0)
                    last_distance = self.distances[threshold_idx]
                    self.distances.pop(0)
                    self.predictions.pop(0)
        return int(sum(self.predictions))


def calculate(rows: list[Bar] | list[TvRow], settings: Settings | None = None, price_scale: float = 0.0) -> list[ResultRow]:
    settings = settings or Settings()
    bars = [row.bar if isinstance(row, TvRow) else row for row in rows]
    close = [bar.close for bar in bars]
    high = [bar.high for bar in bars]
    low = [bar.low for bar in bars]
    open_ = [bar.open for bar in bars]
    hlc3 = [(high[i] + low[i] + close[i]) / 3.0 for i in range(len(bars))]
    ohlc4 = [(open_[i] + high[i] + low[i] + close[i]) / 4.0 for i in range(len(bars))]
    src = select_source(bars, settings.source)

    features = [
        calc_feature(settings.f1, close, high, low, hlc3, price_scale),
        calc_feature(settings.f2, close, high, low, hlc3, price_scale),
        calc_feature(settings.f3, close, high, low, hlc3, price_scale),
        calc_feature(settings.f4, close, high, low, hlc3, price_scale),
        calc_feature(settings.f5, close, high, low, hlc3, price_scale),
    ]
    vol_atr1 = calc_atr(high, low, close, 1)
    vol_atr10 = calc_atr(high, low, close, 10)
    reg_abs_slope, reg_ema_abs_slope = calc_regime_filter(ohlc4, high, low)
    adx_filter = calc_raw_adx_filter(high, low, close, 14)
    ema_filter = ema(close, settings.ema_period)
    sma_filter = sma(close, settings.sma_period)

    yhat1 = [0.0] * len(bars)
    yhat2 = [0.0] * len(bars)
    ann = AnnState()
    direction_buffer = [0] * len(bars)
    buy_buffer = [False] * len(bars)
    sell_buffer = [False] * len(bars)
    results: list[ResultRow] = []

    signal = 0
    bars_held = 0
    bars_since_start_long = 999999
    bars_since_start_short = 999999
    bars_since_alert_bull = 999999
    bars_since_alert_bear = 999999
    prev_signal_change = 0
    prev_signal_change1 = 0
    prev_signal_change2 = 0
    prev_valid_long_exit = False
    prev_valid_short_exit = False
    start_long_price = 0.0
    start_short_price = 0.0
    total_wins = 0
    total_losses = 0
    total_early_flips = 0
    max_bars_back_index = len(bars) - 1 - settings.max_bars_back if len(bars) - 1 >= settings.max_bars_back else 0

    for i, bar in enumerate(bars):
        train_label = 0
        if i >= 4:
            train_label = -1 if src[i - 4] < src[i] else 1 if src[i - 4] > src[i] else 0
        feature_values = tuple(feature[i] for feature in features)  # type: ignore[assignment]
        ann.push(feature_values, train_label)

        yhat1[i] = kernel_rational_quadratic(src, i, settings.kernel_h, settings.kernel_r, settings.kernel_x)
        lag_h = max(settings.kernel_h - settings.kernel_lag, 1)
        yhat2[i] = kernel_gaussian(src, i, lag_h, settings.kernel_x)

        prediction = ann.run(feature_values, settings, len(bars) - 1) if i >= max_bars_back_index else 0

        filt_vol = True
        if settings.use_volatility_filter and not is_missing(vol_atr1[i]) and not is_missing(vol_atr10[i]):
            filt_vol = vol_atr1[i] > vol_atr10[i]
        filt_regime = True
        if settings.use_regime_filter and reg_ema_abs_slope[i] != 0:
            norm_slope = (reg_abs_slope[i] - reg_ema_abs_slope[i]) / reg_ema_abs_slope[i]
            filt_regime = norm_slope >= settings.regime_threshold
        filt_adx = (adx_filter[i] > settings.adx_threshold) if settings.use_adx_filter else True
        filter_all = filt_vol and filt_regime and filt_adx

        is_ema_up = close[i] > ema_filter[i] if settings.use_ema_filter and not is_missing(ema_filter[i]) else not settings.use_ema_filter
        is_ema_down = close[i] < ema_filter[i] if settings.use_ema_filter and not is_missing(ema_filter[i]) else not settings.use_ema_filter
        is_sma_up = close[i] > sma_filter[i] if settings.use_sma_filter and not is_missing(sma_filter[i]) else not settings.use_sma_filter
        is_sma_down = close[i] < sma_filter[i] if settings.use_sma_filter and not is_missing(sma_filter[i]) else not settings.use_sma_filter

        is_bullish_rate = i >= 2 and yhat1[i - 1] < yhat1[i]
        is_bearish_rate = i >= 2 and yhat1[i - 1] > yhat1[i]
        was_bullish_rate = i >= 3 and yhat1[i - 2] < yhat1[i - 1]
        was_bearish_rate = i >= 3 and yhat1[i - 2] > yhat1[i - 1]
        is_bullish_change = is_bullish_rate and was_bearish_rate
        is_bearish_change = is_bearish_rate and was_bullish_rate
        is_bullish_cross = i >= 1 and yhat2[i] >= yhat1[i] and yhat2[i - 1] < yhat1[i - 1]
        is_bearish_cross = i >= 1 and yhat2[i] <= yhat1[i] and yhat2[i - 1] > yhat1[i - 1]
        is_bullish_smooth = yhat2[i] >= yhat1[i]
        is_bearish_smooth = yhat2[i] <= yhat1[i]
        alert_bullish = is_bullish_cross if settings.use_kernel_smoothing else is_bullish_change
        alert_bearish = is_bearish_cross if settings.use_kernel_smoothing else is_bearish_change
        is_bullish = (
            (is_bullish_smooth if settings.use_kernel_smoothing else is_bullish_rate)
            if settings.use_kernel_filter
            else True
        )
        is_bearish = (
            (is_bearish_smooth if settings.use_kernel_smoothing else is_bearish_rate)
            if settings.use_kernel_filter
            else True
        )

        previous_signal = signal
        if prediction > 0 and filter_all:
            signal = 1
        elif prediction < 0 and filter_all:
            signal = -1
        direction_buffer[i] = signal

        signal_change = signal - previous_signal
        is_diff_signal_type = signal_change != 0
        _is_early_flip = is_diff_signal_type and (
            prev_signal_change != 0 or prev_signal_change1 != 0 or prev_signal_change2 != 0
        )
        prev_signal_change2 = prev_signal_change1
        prev_signal_change1 = prev_signal_change
        prev_signal_change = signal_change

        if is_diff_signal_type:
            bars_held = 0
        else:
            bars_held += 1
        is_held_four_bars = bars_held == 4
        is_held_less_than_four_bars = 0 < bars_held < 4

        is_buy = signal == 1 and is_ema_up and is_sma_up
        is_sell = signal == -1 and is_ema_down and is_sma_down
        start_long = is_buy and is_diff_signal_type and is_bullish
        start_short = is_sell and is_diff_signal_type and is_bearish

        is_last_buy = i >= 4 and direction_buffer[i - 4] == 1
        is_last_sell = i >= 4 and direction_buffer[i - 4] == -1

        if start_long:
            bars_since_start_long = 0
        else:
            bars_since_start_long += 1
        if start_short:
            bars_since_start_short = 0
        else:
            bars_since_start_short += 1
        if alert_bullish:
            bars_since_alert_bull = 0
        else:
            bars_since_alert_bull += 1
        if alert_bearish:
            bars_since_alert_bear = 0
        else:
            bars_since_alert_bear += 1

        end_long_strict = (
            (is_held_four_bars and is_last_buy) or (is_held_less_than_four_bars and start_short and is_last_buy)
        ) and i >= 4 and buy_buffer[i - 4]
        end_short_strict = (
            (is_held_four_bars and is_last_sell) or (is_held_less_than_four_bars and start_long and is_last_sell)
        ) and i >= 4 and sell_buffer[i - 4]
        is_valid_long_exit = bars_since_alert_bear > bars_since_start_long
        is_valid_short_exit = bars_since_alert_bull > bars_since_start_short
        end_long_dynamic = is_bearish_change and prev_valid_long_exit
        end_short_dynamic = is_bullish_change and prev_valid_short_exit
        dynamic_valid = not settings.use_ema_filter and not settings.use_sma_filter and not settings.use_kernel_smoothing
        end_long = end_long_dynamic if settings.use_dynamic_exits and dynamic_valid else end_long_strict
        end_short = end_short_dynamic if settings.use_dynamic_exits and dynamic_valid else end_short_strict

        buy_buffer[i] = start_long
        sell_buffer[i] = start_short
        prev_valid_long_exit = is_valid_long_exit
        prev_valid_short_exit = is_valid_short_exit

        market_price = src[i] if settings.use_worst_case else (high[i] + low[i] + open_[i] + open_[i]) / 4.0
        if i > max_bars_back_index:
            early_flips = 0
            wins = 0
            losses = 0
            if start_long:
                start_short_price = 0.0
                early_flips = 1 if _is_early_flip else 0
                start_long_price = market_price
            if end_long:
                delta = market_price - start_long_price
                wins = 1 if delta > 0 else 0
                losses = 1 if delta < 0 else 0
            if start_short:
                start_long_price = 0.0
                start_short_price = market_price
            if end_short:
                early_flips = 1 if _is_early_flip else early_flips
                delta = start_short_price - market_price
                wins = 1 if delta > 0 else wins
                losses = 1 if delta < 0 else losses
            total_wins += wins
            total_losses += losses
            total_early_flips += early_flips

        total_trades = total_wins + total_losses
        win_loss_ratio = total_wins / total_trades if total_trades else math.nan
        table_wl_ratio = total_wins / total_losses if total_losses else math.nan
        win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) else math.nan
        backtest_stream = 1 if start_long else 2 if end_long else -1 if start_short else -2 if end_short else 0
        stop_buy = end_long and settings.show_exits
        stop_sell = end_short and settings.show_exits
        c_green = pine_color("#009988", 20)
        c_red = pine_color("#CC3311", 20)
        transparent = pine_color("#000000", 100)
        kernel_bullish = is_bullish_smooth if settings.use_kernel_smoothing else is_bullish_rate
        kernel_plot_color = (c_green if kernel_bullish else c_red) if settings.show_kernel_estimate else transparent
        neutral_color = pine_color("#787b86", 25)
        if prediction > 0:
            prediction_color = prediction_green(prediction, settings.use_confidence_gradient)
        elif prediction < 0:
            prediction_color = prediction_red(-prediction, settings.use_confidence_gradient)
        else:
            prediction_color = neutral_color
        prediction_label_color = prediction_color if settings.show_bar_predictions else ""
        bar_color = (
            color_with_transparency(prediction_color, 50 if settings.use_confidence_gradient else 30)
            if settings.show_bar_colors
            else ""
        )
        if settings.use_atr_offset:
            prediction_label_y = high[i] + vol_atr1[i] if prediction > 0 else low[i] - vol_atr1[i]
        else:
            hl2 = (high[i] + low[i]) / 2.0
            prediction_label_y = (
                high[i] + hl2 * settings.bar_predictions_offset / 20.0
                if prediction > 0
                else low[i] - hl2 * settings.bar_predictions_offset / 30.0
            )
        results.append(
            ResultRow(
                bar=bar,
                f1=features[0][i],
                f2=features[1][i],
                f3=features[2][i],
                f4=features[3][i],
                f5=features[4][i],
                kernel=yhat1[i],
                prediction=prediction,
                direction=signal,
                buy=start_long,
                sell=start_short,
                exit_buy=end_long,
                exit_sell=end_short,
                stop_buy=stop_buy,
                stop_sell=stop_sell,
                backtest_stream=backtest_stream,
                open_long_alert=start_long,
                close_long_alert=end_long,
                open_short_alert=start_short,
                close_short_alert=end_short,
                open_position_alert=start_long or start_short,
                close_position_alert=end_long or end_short,
                kernel_bullish_alert=alert_bullish,
                kernel_bearish_alert=alert_bearish,
                kernel_plot_color=kernel_plot_color,
                prediction_label=str(prediction),
                prediction_label_y=prediction_label_y,
                prediction_label_color=prediction_label_color,
                bar_color=bar_color,
                trade_stats_visible=settings.show_trade_stats,
                trade_stats_header="\U0001f4c8 Trade Stats",
                total_wins=total_wins,
                total_losses=total_losses,
                total_early_signal_flips=total_early_flips,
                total_trades=total_trades,
                win_loss_ratio=win_loss_ratio,
                table_wl_ratio=table_wl_ratio,
                win_rate=win_rate,
            )
        )
    return results
