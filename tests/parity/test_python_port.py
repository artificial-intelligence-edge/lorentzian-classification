from __future__ import annotations

import os
from pathlib import Path
import csv
import contextlib
from dataclasses import fields, replace
import io
import json
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_PORT = ROOT / "ports" / "python"
sys.path.insert(0, str(PYTHON_PORT))

from lorentzian_classification import (  # noqa: E402
    Bar,
    LorentzianClassification,
    ResultRow,
    Settings,
    TvRow,
    __version__,
    calculate,
    result_to_mapping,
    rows_from_records,
    read_tradingview_csv,
)
from lorentzian_classification.cli import (  # noqa: E402
    EXPORT_PACK_HEADER,
    FIXTURE_DIR_ENV_VAR,
    MINIMUM_PINE_EXPORT_COLUMNS,
    PINE_EXPORT_SERIES,
    PARITY_FIXTURES,
    RESULT_FIELDNAMES,
    SETTINGS_FINGERPRINT_COLUMN,
    build_parser,
    classify_csv_candidate,
    file_sha256,
    fixture_dir_from_arg,
    fixture_search_dirs_from_arg,
    full_numeric_export_columns_for_settings,
    has_debug_markers,
    load_manifest_payload,
    load_external_parity_report_specs,
    load_external_source_specs,
    load_parity_manifest,
    load_pine_source_specs,
    external_compiled_artifact_records,
    external_indicator_input_contract_records,
    external_parity_script_contract_records,
    external_parity_report_records,
    externalParityReportSpec,
    external_source_sanity_records,
    pine_export_schema,
    pine_export_columns_for_settings,
    feature_export_columns,
    expected_pine_input_contract_rows,
    pine_export_series_for_settings,
    pine_source_records,
    parity_summary,
    required_settings_smoke_records,
    settings_from_mapping,
    settings_from_args,
    settings_fingerprint,
    source_code_sha256,
    source_records,
    validate_source_records,
    write_results,
)


DEFAULT_FIXTURE_DIR = Path(
    os.environ.get("LORENTZIAN_PARITY_FIXTURE_DIR")
    or ROOT / "tests" / "parity" / "baselines"
)
PINE_LCV6 = DEFAULT_FIXTURE_DIR.parent / "PineScript" / "Indicators" / "lcv6.pine"
PINE_ML_EXTENSIONS = DEFAULT_FIXTURE_DIR.parent / "PineScript" / "Libraries" / "MLExtensions.pine"
PINE_KERNEL_FUNCTIONS = DEFAULT_FIXTURE_DIR.parent / "PineScript" / "Libraries" / "KernelFunctions.pine"
external_INDICATOR = DEFAULT_FIXTURE_DIR.parent / "Indicators" / "LorentzianClassification" / "LorentzianClassification.external_src"
PYTHON_CORE = PYTHON_PORT / "lorentzian_classification" / "core.py"
REPO_PINE_LCV6 = ROOT / "ports" / "pinescript" / "lorentzian-classification-v2.pine"
REPO_PINE_ML_EXTENSIONS = ROOT / "ports" / "pinescript" / "libraries" / "MLExtensions.pine"
REPO_PINE_KERNEL_FUNCTIONS = ROOT / "ports" / "pinescript" / "libraries" / "KernelFunctions.pine"
BASELINES_DIR = ROOT / "tests" / "parity" / "baselines"
BASELINES_MANIFEST = BASELINES_DIR / "baselines_manifest.json"


def write_full_export_fixture(
    path: Path,
    settings: Settings,
    fingerprint: int | None = None,
    include_fingerprint: bool = True,
) -> list[str]:
    columns = [
        str(row["column"])
        for row in pine_export_series_for_settings(settings, include_full=True)
        if row["export_mode"] != "encoded_helper_required"
    ]
    if include_fingerprint:
        columns.append(SETTINGS_FINGERPRINT_COLUMN)
    row_values = {column: "" for column in columns}
    row_values.update(
        {
            "time": "2026-01-01",
            "open": "1.10",
            "high": "1.20",
            "low": "1.00",
            "close": "1.15",
            "Prediction": "0",
            "Direction": "0",
            "Backtest Stream": "0",
            "Kernel Plot Color": "-120",
            "Prediction Label": "0",
            "Prediction Label Color": "225",
            "Bar Color": "250",
            "Trade Stats Visible": "1",
            "Trade Stats Header": "1",
        }
    )
    if include_fingerprint:
        row_values[SETTINGS_FINGERPRINT_COLUMN] = str(
            settings_fingerprint(settings) if fingerprint is None else fingerprint
        )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow([row_values[column] for column in columns])
    return columns


def parse_pine_literal(value: str):
    normalized = value.strip().rstrip(".")
    if normalized in {"true", "false"}:
        return normalized == "true"
    if normalized in {"close", "open", "high", "low", "hl2", "hlc3", "ohlc4"}:
        return normalized
    if (normalized.startswith('"') and normalized.endswith('"')) or (
        normalized.startswith("'") and normalized.endswith("'")
    ):
        return normalized[1:-1]
    if "." in normalized:
        return float(normalized)
    return int(normalized)


def pine_defval_for_title(source: str, title: str):
    escaped = re.escape(title)
    pattern = rf"input\.\w+\(.*?title\s*=\s*['\"]{escaped}['\"].*?defval\s*=\s*([^,\)]+)"
    match = re.search(pattern, source)
    if not match:
        raise AssertionError(f"could not find Pine input default for title {title!r}")
    return parse_pine_literal(match.group(1))


def pine_defval_for_inline(source: str, title: str, inline: str):
    escaped_title = re.escape(title)
    escaped_inline = re.escape(inline)
    pattern = (
        rf"input\.\w+\(.*?title\s*=\s*['\"]{escaped_title}['\"].*?"
        rf"defval\s*=\s*([^,\)]+).*?inline\s*=\s*['\"]{escaped_inline}['\"]"
    )
    match = re.search(pattern, source)
    if not match:
        raise AssertionError(f"could not find Pine input default for title {title!r} inline {inline!r}")
    return parse_pine_literal(match.group(1))


def pine_defval_for_variable(source: str, variable: str):
    pattern = rf"^{re.escape(variable)}\s*=\s*input\.\w+\(.*?defval\s*=\s*([^,\)]+)"
    match = re.search(pattern, source, re.MULTILINE)
    if not match:
        raise AssertionError(f"could not find Pine input default for variable {variable!r}")
    return parse_pine_literal(match.group(1))


def pine_positional_default(source: str, variable: str):
    pattern = rf"^{re.escape(variable)}\s*=\s*input\.\w+\(([^,\)]+)"
    match = re.search(pattern, source, re.MULTILINE)
    if not match:
        raise AssertionError(f"could not find Pine positional input default for variable {variable!r}")
    return parse_pine_literal(match.group(1))


def pine_input_identifiers(source: str) -> set[str]:
    identifiers: set[str] = set()
    for line in source.splitlines():
        if "input." not in line:
            continue
        variable_match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*input\.", line)
        if variable_match:
            identifiers.add(variable_match.group(1))
            continue
        title_match = re.search(r"title\s*=\s*['\"]([^'\"]+)['\"]", line)
        if not title_match:
            continue
        title = title_match.group(1)
        inline_match = re.search(r"inline\s*=\s*['\"]([^'\"]+)['\"]", line)
        if title in {"Threshold", "Period"} and inline_match:
            identifiers.add(f"{title}|{inline_match.group(1)}")
        else:
            identifiers.add(title)
    return identifiers


def pine_call_records(source: str, call: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith(f"{call}("):
            continue
        title_match = re.search(r"\btitle\s*=\s*(['\"])(.*?)\1", stripped)
        if title_match:
            title = title_match.group(2)
        else:
            title_match = re.search(r"(['\"])(.*?)\1", stripped)
            if not title_match:
                continue
            title = title_match.group(2)

        inner = stripped[len(call) + 1 : stripped.rfind(")")]
        condition_match = re.search(r"\bcondition\s*=\s*([^,]+)", inner)
        expression = condition_match.group(1).strip() if condition_match else inner.split(",", 1)[0].strip()
        records[title] = expression
    return records


def pine_condition_head(expression: str) -> str:
    return expression.split("?", 1)[0].strip()


def pine_switch_cases(source: str, variable: str) -> dict[str, str]:
    cases: dict[str, str] = {}
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"{variable} = switch")
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        match = re.match(r"(.+?)\s*=>\s*(-?\d+)\s*$", stripped)
        if not match:
            break
        cases[match.group(1).strip()] = match.group(2)
    return cases


def output_signature(results: list[ResultRow]) -> dict[str, object]:
    last = results[-1]

    def rounded(value: float) -> float | None:
        return None if value != value else round(value, 8)

    return {
        "prediction_sum": sum(row.prediction for row in results),
        "direction_sum": sum(row.direction for row in results),
        "buy_count": sum(row.buy for row in results),
        "sell_count": sum(row.sell for row in results),
        "exit_buy_count": sum(row.exit_buy for row in results),
        "exit_sell_count": sum(row.exit_sell for row in results),
        "stop_buy_count": sum(row.stop_buy for row in results),
        "stop_sell_count": sum(row.stop_sell for row in results),
        "last_kernel": rounded(last.kernel),
        "total_wins": last.total_wins,
        "total_losses": last.total_losses,
        "total_trades": last.total_trades,
        "win_loss_ratio": rounded(last.win_loss_ratio),
        "trade_stats_visible": last.trade_stats_visible,
        "last_f1": rounded(last.f1),
        "last_f2": rounded(last.f2),
        "last_f3": rounded(last.f3),
        "last_f4": rounded(last.f4),
        "last_f5": rounded(last.f5),
    }


def pine_contract_literal(value: object, setting: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value if setting == "source" else json.dumps(value)
    return str(value)


def pine_contract_source(*, neighbors_count: int = 8) -> str:
    lines = ["// synthetic lcv6 input surface for CLI contract tests"]
    for row in expected_pine_input_contract_rows():
        setting = str(row["setting"])
        value = neighbors_count if setting == "neighbors_count" else row["value"]
        if setting == "source":
            kind = "source"
        elif setting in {"f1", "f2", "f3", "f4", "f5"} and str(row.get("variable", "")).endswith("_string"):
            kind = "string"
        elif isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, int):
            kind = "int"
        elif isinstance(value, float):
            kind = "float"
        else:
            kind = "string"
        args = [
            f"title={json.dumps(row['title'])}",
            f"defval={pine_contract_literal(value, setting)}",
            f"group={json.dumps(row['group'])}",
        ]
        inline = row.get("inline")
        if inline:
            args.append(f"inline={json.dumps(inline)}")
        call = f"input.{kind}({', '.join(args)})"
        variable = row.get("variable")
        lines.append(f"{variable} = {call}" if variable else call)
    return "\n".join(lines) + "\n"


def pine_output_contract_source(*, buy_condition: str = "startLongTrade") -> str:
    return f"""
plot(kernelEstimate, color=plotColor, linewidth=2, title="Kernel Regression Estimate")
plotshape({buy_condition} ? low : na, 'Buy', shape.labelup, location.belowbar)
plotshape(startShortTrade ? high : na, 'Sell', shape.labeldown, location.abovebar)
plotshape(endLongTrade and settings.showExits ? high : na, 'StopBuy', shape.xcross, location.absolute)
plotshape(endShortTrade and settings.showExits ? low : na, 'StopSell', shape.xcross, location.absolute)
alertcondition(startLongTrade, title='Open Long ▲', message='open')
alertcondition(endLongTrade, title='Close Long ▲', message='close')
alertcondition(startShortTrade, title='Open Short ▼', message='open')
alertcondition(endShortTrade, title='Close Short ▼', message='close')
alertcondition(startShortTrade or startLongTrade, title='Open Position ▲▼', message='open')
alertcondition(endShortTrade or endLongTrade, title='Close Position ▲▼', message='close')
alertcondition(condition=alertBullish, title='Kernel Bullish Color Change', message='bull')
alertcondition(condition=alertBearish, title='Kernel Bearish Color Change', message='bear')
x_val = bar_index
y_val = high
c_label = color.gray
label.new(x_val, y_val, str.tostring(prediction), xloc.bar_index, yloc.price)
barcolor(showBarColors ? c_bars : na)
backTestStream = switch
    startLongTrade => 1
    endLongTrade => 2
    startShortTrade => -1
    endShortTrade => -2
plot(backTestStream, "Backtest Stream", display=display.none)
[totalWins, totalLosses, totalEarlySignalFlips, totalTrades, tradeStatsHeader, winLossRatio, winRate] = ml.backtest()
if showTradeStats
    table.cell(tbl, 0, 0, tradeStatsHeader)
    table.cell(tbl, 1, 1, str.tostring(totalWins / totalTrades, '#.#%'))
    table.cell(tbl, 1, 2, str.tostring(totalTrades, '#') + ' (' + str.tostring(totalWins, '#') + '|' + str.tostring(totalLosses, '#') + ')')
    table.cell(tbl, 1, 5, str.tostring(totalWins / totalLosses, '0.00'))
    table.cell(tbl, 1, 6, str.tostring(totalEarlySignalFlips, '#'))
"""


def external_parity_script_contract_source(*, include_output_input: bool = True) -> str:
    output_input = 'input string InpOutputFile = "parity.csv";\n' if include_output_input else ""
    return f"""
#property script_show_inputs
input string InpInputFile = "input.csv";
{output_input}input bool InpIncludeFullHist = false;
int ReadCSV(string filename) {{ return 0; }}
void OnStart()
{{
   int bars = ReadCSV(InpInputFile);
   int out = FileOpen(InpOutputFile, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(InpIncludeFullHist) {{ bars = bars + 0; }}
   FileWrite(out, "time", "close",
             "tv_f1", "external_f1", "f1_diff",
             "tv_f2", "external_f2", "f2_diff",
             "tv_f3", "external_f3", "f3_diff",
             "tv_f4", "external_f4", "f4_diff",
             "tv_f5", "external_f5", "f5_diff",
             "tv_kernel", "external_kernel", "kernel_diff",
             "tv_prediction", "external_prediction", "pred_match",
             "tv_direction", "external_direction", "dir_match",
             "tv_buy", "external_buy", "tv_sell", "external_sell");
}}
"""


def external_indicator_input_contract_source(
    *, use_atr_offset: str = "true", include_confidence_gradient: bool = True
) -> str:
    confidence = "input bool   InpUseConfidenceGradient = true;\n" if include_confidence_gradient else ""
    return f"""
input ENUM_APPLIED_PRICE InpSource = PRICE_CLOSE;
input int    InpNeighborsCount   = 8;
input int    InpMaxBarsBack      = 2000;
input int    InpFeatureCount     = 5;
input int    InpColorCompression = 1;
input bool   InpShowExits        = false;
input bool   InpUseDynamicExits  = false;
input bool   InpIncludeFullHist  = false;
input bool   InpShowTradeStats   = true;
input bool   InpUseWorstCase     = false;
input bool   InpUseVolFilter     = true;
input bool   InpUseRegimeFilter  = true;
input bool   InpUseAdxFilter     = false;
input double InpRegimeThreshold  = -0.1;
input int    InpAdxThreshold     = 20;
input bool   InpUseEmaFilter     = false;
input int    InpEmaPeriod        = 200;
input bool   InpUseSmaFilter     = false;
input int    InpSmaPeriod        = 200;
input bool   InpUseKernelFilter    = true;
input bool   InpShowKernelEst      = true;
input bool   InpUseKernelSmoothing = false;
input int    InpKernelH            = 8;
input double InpKernelR            = 8.0;
input int    InpKernelX            = 25;
input int    InpKernelLag          = 2;
input ENUM_FEATURE_TYPE InpF1Type = FEATURE_RSI;
input int    InpF1ParamA = 14;
input int    InpF1ParamB = 1;
input ENUM_FEATURE_TYPE InpF2Type = FEATURE_WT;
input int    InpF2ParamA = 10;
input int    InpF2ParamB = 11;
input ENUM_FEATURE_TYPE InpF3Type = FEATURE_CCI;
input int    InpF3ParamA = 20;
input int    InpF3ParamB = 1;
input ENUM_FEATURE_TYPE InpF4Type = FEATURE_ADX;
input int    InpF4ParamA = 20;
input int    InpF4ParamB = 2;
input ENUM_FEATURE_TYPE InpF5Type = FEATURE_RSI;
input int    InpF5ParamA = 9;
input int    InpF5ParamB = 1;
input bool   InpShowBarColors     = true;
input bool   InpShowBarPreds      = true;
input bool   InpUseAtrOffset      = {use_atr_offset};
input double InpBarPredOffset     = 0;
{confidence}
int begin = 0;
bool filterAll = filtVol && filtRegime && filtAdx;
"""


class PythonPortParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_dir = Path(os.environ.get("LORENTZIAN_PARITY_FIXTURE_DIR", DEFAULT_FIXTURE_DIR))
        if not cls.fixture_dir.exists():
            raise unittest.SkipTest(f"parity fixture directory not found: {cls.fixture_dir}")

    def assert_parity(self, filename: str, *, include_full_history: bool, tolerance: float = 1e-9) -> None:
        path = self.fixture_dir / filename
        self.assertTrue(path.exists(), f"missing parity fixture: {path}")
        tv_rows, price_scale = read_tradingview_csv(path)
        results = calculate(
            tv_rows,
            settings=Settings(include_full_history=include_full_history),
            price_scale=price_scale,
        )
        summary, mismatches = parity_summary(tv_rows, results, tolerance)
        self.assertTrue(summary["pass"], f"{filename} failed parity: {summary}; mismatches={mismatches[:5]}")

    def test_oanda_daily_full_history(self) -> None:
        self.assert_parity("pine_oanda_eurusd_1d_full_history.csv", include_full_history=True)

    def test_tastyfx_daily_full_history(self) -> None:
        self.assert_parity("pine_tastyfx_eurusd_1d_full_history.csv", include_full_history=True)

    def test_coinbase_daily_limited_history(self) -> None:
        self.assert_parity("pine_coinbase_btcusd_1d_limited_history.csv", include_full_history=False)

    def test_btc_h1_trimmed_window_limited_history(self) -> None:
        self.assert_parity("pine_btcusd_h1_trimmed_limited_history.csv", include_full_history=False)

    def test_cli_run_emits_business_outputs(self) -> None:
        input_path = self.fixture_dir / "pine_coinbase_btcusd_1d_limited_history.csv"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "python_output.csv"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "run",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                check=True,
            )
            with output_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(len(rows), 4155)
            self.assertIn("Backtest Stream", reader.fieldnames or [])
            self.assertIn("Open Long Alert", reader.fieldnames or [])
            self.assertIn("Close Long Alert", reader.fieldnames or [])
            self.assertIn("Open Short Alert", reader.fieldnames or [])
            self.assertIn("Close Short Alert", reader.fieldnames or [])
            self.assertIn("Open Position Alert", reader.fieldnames or [])
            self.assertIn("Close Position Alert", reader.fieldnames or [])
            self.assertIn("Kernel Bullish Alert", reader.fieldnames or [])
            self.assertIn("Kernel Bearish Alert", reader.fieldnames or [])
            self.assertIn("Kernel Plot Color", reader.fieldnames or [])
            self.assertIn("Prediction Label", reader.fieldnames or [])
            self.assertIn("Prediction Label Y", reader.fieldnames or [])
            self.assertIn("Prediction Label Color", reader.fieldnames or [])
            self.assertIn("Bar Color", reader.fieldnames or [])
            self.assertIn("Trade Stats Visible", reader.fieldnames or [])
            self.assertIn("Trade Stats Header", reader.fieldnames or [])
            self.assertIn("Total Wins", reader.fieldnames or [])
            self.assertIn("Total Losses", reader.fieldnames or [])
            self.assertIn("Total Early Signal Flips", reader.fieldnames or [])
            self.assertIn("Table WL Ratio", reader.fieldnames or [])
            self.assertEqual(rows[-1]["Prediction"], "-4")
            self.assertEqual(rows[-1]["Direction"], "1")
            self.assertTrue(rows[-1]["Total Trades"])
            if PINE_LCV6.exists():
                source = PINE_LCV6.read_text()
                alert_column_map = {
                    "Open Long": "Open Long Alert",
                    "Close Long": "Close Long Alert",
                    "Open Short": "Open Short Alert",
                    "Close Short": "Close Short Alert",
                    "Open Position": "Open Position Alert",
                    "Close Position": "Close Position Alert",
                    "Kernel Bullish Color Change": "Kernel Bullish Alert",
                    "Kernel Bearish Color Change": "Kernel Bearish Alert",
                }
                for title, column in alert_column_map.items():
                    self.assertIn(title, source)
                    self.assertIn(column, reader.fieldnames or [])
                output_surface_map = {
                    "Kernel Regression Estimate": "Kernel Regression Estimate",
                    "'Buy'": "Buy",
                    "'Sell'": "Sell",
                    "'StopBuy'": "StopBuy",
                    "'StopSell'": "StopSell",
                    "label.new": "Prediction Label",
                    "y_val": "Prediction Label Y",
                    "c_label": "Prediction Label Color",
                    "barcolor": "Bar Color",
                    "showTradeStats": "Trade Stats Visible",
                    "Backtest Stream": "Backtest Stream",
                    "tradeStatsHeader": "Trade Stats Header",
                    "totalWins": "Total Wins",
                    "totalLosses": "Total Losses",
                    "totalEarlySignalFlips": "Total Early Signal Flips",
                    "totalTrades": "Total Trades",
                    "winLossRatio": "Win Loss Ratio",
                    "totalWins / totalLosses": "Table WL Ratio",
                    "winRate": "Win Rate",
                }
                for pine_token, column in output_surface_map.items():
                    self.assertIn(pine_token, source)
                    self.assertIn(column, reader.fieldnames or [])

    def test_library_wrapper_matches_direct_calculation_for_csv_path(self) -> None:
        input_path = self.fixture_dir / "pine_coinbase_btcusd_1d_limited_history.csv"
        tv_rows, price_scale = read_tradingview_csv(input_path)
        expected = calculate(tv_rows, settings=Settings(), price_scale=price_scale)

        model = LorentzianClassification(input_path)

        self.assertEqual(model.rows, tv_rows)
        self.assertEqual(model.price_scale, price_scale)
        self.assertEqual(model.results, expected)
        self.assertEqual(model.latest, expected[-1])
        self.assertEqual(model.data, expected)

    def test_library_settings_from_mapping_matches_issue_example(self) -> None:
        settings = Settings.from_mapping(
            {
                "source": "close",
                "neighbors_count": 12,
                "max_bars_back": 1000,
                "use_adx_filter": True,
                "f1": "RSI:14:1",
                "f2": "WT:10:11",
                "f3": "CCI:20:1",
                "f4": "ADX:20:2",
                "f5": "RSI:9:1",
            }
        )

        self.assertEqual(settings.neighbors_count, 12)
        self.assertEqual(settings.max_bars_back, 1000)
        self.assertTrue(settings.use_adx_filter)
        self.assertEqual(settings.f1, ("RSI", 14, 1))
        self.assertEqual(settings.f2, ("WT", 10, 11))
        self.assertEqual(settings.f3, ("CCI", 20, 1))
        self.assertEqual(settings.f4, ("ADX", 20, 2))
        self.assertEqual(settings.f5, ("RSI", 9, 1))
        self.assertEqual(settings.to_mapping()["neighbors_count"], 12)

        input_path = self.fixture_dir / "pine_coinbase_btcusd_1d_limited_history.csv"
        model = LorentzianClassification(input_path, settings=settings)
        self.assertIs(model.settings, settings)
        self.assertEqual(len(model.results), 4155)

    def test_library_settings_validate_at_construction(self) -> None:
        invalid_settings = [
            ({"source": "median"}, "source: must be one of"),
            ({"neighbors_count": 0}, "neighbors_count: must be >= 1"),
            ({"include_full_history": "true"}, "include_full_history: must be a boolean"),
            ({"f1": ("UNKNOWN", 14, 1)}, "f1: feature type must be one of"),
        ]
        for overrides, message in invalid_settings:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    Settings(**overrides)

        with self.assertRaisesRegex(ValueError, "unknown settings keys: neighbor_count"):
            Settings.from_mapping({"neighbor_count": 12})

    def test_library_wrapper_accepts_mapping_records_and_dataframe_like_input(self) -> None:
        input_path = self.fixture_dir / "pine_coinbase_btcusd_1d_limited_history.csv"
        with input_path.open(newline="") as handle:
            records = list(csv.DictReader(handle))[:80]
        tv_rows, price_scale = rows_from_records(records)
        expected = calculate(tv_rows, settings=Settings(), price_scale=price_scale)

        self.assertEqual(LorentzianClassification(records).results, expected)

        class FrameLike:
            columns = ["open", "high", "low", "close"]

            def __init__(self, rows: list[dict[str, str]], index: list[str]) -> None:
                self._rows = rows
                self.index = index

            def to_dict(self, orient: str):
                if orient != "records":
                    raise ValueError(f"unsupported orient: {orient}")
                return self._rows

        frame_rows = [{column: row[column] for column in FrameLike.columns} for row in records]
        frame = FrameLike(frame_rows, [row["time"] for row in records])
        frame_model = LorentzianClassification(frame)
        self.assertEqual([row.bar.time for row in frame_model.rows], [row["time"] for row in records])
        self.assertEqual(frame_model.results, expected)

    def test_library_wrapper_dump_uses_cli_result_schema(self) -> None:
        input_path = self.fixture_dir / "pine_coinbase_btcusd_1d_limited_history.csv"
        model = LorentzianClassification(input_path)
        with tempfile.TemporaryDirectory() as tmp:
            cli_output = Path(tmp) / "cli.csv"
            library_output = Path(tmp) / "library.csv"
            write_results(cli_output, model.results)
            self.assertEqual(model.dump(library_output), library_output)

            self.assertEqual(library_output.read_bytes(), cli_output.read_bytes())
            with library_output.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, RESULT_FIELDNAMES)
            self.assertEqual(rows[-1], {key: str(value) for key, value in result_to_mapping(model.latest).items()})

    def test_library_wrapper_rejects_missing_mapping_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing high column"):
            LorentzianClassification([{"time": "2026-01-01", "open": "1", "low": "1", "close": "1"}])

    def test_python_package_exposes_library_api(self) -> None:
        import lorentzian_classification as package

        expected_exports = {
            "LorentzianClassification",
            "RESULT_FIELDNAMES",
            "Settings",
            "calculate",
            "read_tradingview_csv",
            "rows_from_records",
            "result_to_mapping",
            "write_result_csv",
        }
        self.assertTrue(expected_exports.issubset(set(package.__all__)))
        self.assertIs(package.LorentzianClassification, LorentzianClassification)
        self.assertIs(package.Settings, Settings)
        self.assertTrue((PYTHON_PORT / "lorentzian_classification" / "py.typed").is_file())

        pyproject = tomllib.loads((PYTHON_PORT / "pyproject.toml").read_text())
        self.assertEqual(pyproject["project"]["optional-dependencies"]["dataframe"], ["pandas>=2"])
        self.assertIn("py.typed", pyproject["tool"]["setuptools"]["package-data"]["lorentzian_classification"])

    def test_python_readme_documents_validated_library_settings(self) -> None:
        readme = (PYTHON_PORT / "README.md").read_text()

        self.assertIn("Settings.from_mapping({", readme)
        self.assertIn('neighbors_count=12', readme)
        self.assertIn('f1=("RSI", 14, 1)', readme)
        self.assertIn('"f1": "RSI:14:1"', readme)
        self.assertIn("Direct `Settings(...)` construction uses the same", readme)
        self.assertIn("Reserved for tuple compatibility", readme)
        for field in fields(Settings):
            self.assertIn(f"`{field.name}`", readme)

    def test_cli_run_rejects_header_only_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "header_only.csv"
            output_path = Path(tmp) / "python_output.csv"
            input_path.write_text("time,open,high,low,close\n")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "run",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            output_exists = output_path.exists()

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid input CSV", completed.stdout)
        self.assertIn("no data rows", completed.stdout)
        self.assertFalse(output_exists)

    def test_cli_run_rejects_duplicate_input_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "duplicate_columns.csv"
            output_path = Path(tmp) / "python_output.csv"
            input_path.write_text(
                "\n".join(
                    [
                        "time,open,open,high,low,close",
                        "2026-01-01,1.10,1.11,1.20,1.00,1.15",
                    ]
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "run",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            output_exists = output_path.exists()

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid input CSV", completed.stdout)
        self.assertIn("duplicate input columns:", completed.stdout)
        self.assertIn("open", completed.stdout)
        self.assertFalse(output_exists)

    def test_cli_run_rejects_missing_input_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "missing_close.csv"
            output_path = Path(tmp) / "python_output.csv"
            input_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low",
                        "2026-01-01,1.10,1.20,1.00",
                    ]
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "run",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            output_exists = output_path.exists()

        self.assertEqual(completed.returncode, 1)
        self.assertIn("missing required input columns:", completed.stdout)
        self.assertIn("close", completed.stdout)
        self.assertFalse(output_exists)

    def test_cli_run_rejects_missing_input_file_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "missing.csv"
            output_path = Path(tmp) / "python_output.csv"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "run",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            output_exists = output_path.exists()

        self.assertEqual(completed.returncode, 1)
        self.assertIn("input file not found", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)
        self.assertFalse(output_exists)

    def test_cli_run_rejects_directory_input_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input_dir"
            output_path = Path(tmp) / "python_output.csv"
            input_path.mkdir()

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "run",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            output_exists = output_path.exists()

        self.assertEqual(completed.returncode, 1)
        self.assertIn("input path is not a file", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)
        self.assertFalse(output_exists)

    def test_cli_run_rejects_missing_output_directory_without_traceback(self) -> None:
        input_path = self.fixture_dir / "pine_coinbase_btcusd_1d_limited_history.csv"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "missing" / "python_output.csv"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "run",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            output_exists = output_path.exists()

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid output path", completed.stdout)
        self.assertIn("output directory not found", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)
        self.assertFalse(output_exists)

    def test_python_package_exposes_all_cli_commands(self) -> None:
        expected_commands = {
            "run",
            "parity",
            "validate-fixtures",
            "audit-fixtures",
            "pine-input-contract",
            "pine-output-contract",
            "external-report-checklist",
            "external-runner-pack",
            "verify-external-runner-pack",
            "prepare-readiness-artifacts",
            "verify-readiness-artifacts",
            "export-checklist",
            "import-pine-exports",
            "export-pack",
            "verify-export-pack",
            "pine-export-helper",
            "readiness",
            "readiness-blockers",
        }
        pyproject = tomllib.loads((PYTHON_PORT / "pyproject.toml").read_text())
        self.assertEqual(
            pyproject["project"]["scripts"]["lorentzian-classification"],
            "lorentzian_classification.cli:main",
        )
        self.assertEqual(pyproject["project"]["version"], __version__)
        self.assertEqual(pyproject["project"]["readme"], "README.md")
        self.assertGreaterEqual(pyproject["project"]["requires-python"], ">=3.10")
        self.assertEqual(
            pyproject["tool"]["setuptools"]["packages"]["find"]["include"],
            ["lorentzian_classification*"],
        )

        parser = build_parser()
        subparser_actions = [
            action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
        ]
        self.assertEqual(len(subparser_actions), 1)
        self.assertEqual(set(subparser_actions[0].choices), expected_commands)

        completed = subprocess.run(
            [sys.executable, "-m", "lorentzian_classification", "--help"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("usage: lorentzian-classification", completed.stdout)
        for command in expected_commands:
            self.assertIn(command, completed.stdout)

        completed = subprocess.run(
            [sys.executable, "-m", "lorentzian_classification", "--version"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), f"lorentzian-classification {__version__}")

    def test_pine_input_contract_reports_cli_setting_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            source_path = root / "PineScript" / "Indicators" / "lcv6.pine"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(pine_contract_source())
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "contract test source",
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "pine-input-contract",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["summary"]["contracts_failed"], 0)
        self.assertEqual(payload["contracts"][0]["actual_count"], len(expected_pine_input_contract_rows()))
        self.assertEqual(payload["contracts"][0]["expected_count"], len(expected_pine_input_contract_rows()))

    def test_pine_input_contract_blocks_pine_default_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            source_path = root / "PineScript" / "Indicators" / "lcv6.pine"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(pine_contract_source(neighbors_count=9))
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "contract drift test source",
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "pine-input-contract",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["summary"]["contracts_failed"], 1)
        mismatch = payload["contracts"][0]["mismatches"][0]
        self.assertEqual(mismatch["setting"], "neighbors_count")
        self.assertEqual(mismatch["field"], "value")
        self.assertEqual(mismatch["expected"], 8)
        self.assertEqual(mismatch["actual"], 9)

    def test_pine_output_contract_reports_cli_output_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            source_path = root / "PineScript" / "Indicators" / "lcv6.pine"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(pine_output_contract_source())
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "output contract test source",
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "pine-output-contract",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["summary"]["contracts_failed"], 0)
        self.assertEqual(payload["contracts"][0]["missing"], [])
        self.assertEqual(payload["contracts"][0]["mismatches"], [])

    def test_pine_output_contract_blocks_plot_condition_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            source_path = root / "PineScript" / "Indicators" / "lcv6.pine"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(pine_output_contract_source(buy_condition="changedLongTrade"))
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "output contract drift test source",
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "pine-output-contract",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["passed"])
        mismatch = payload["contracts"][0]["mismatches"][0]
        self.assertEqual(mismatch["kind"], "plotshape")
        self.assertEqual(mismatch["title"], "Buy")
        self.assertEqual(mismatch["column"], "Buy")
        self.assertEqual(mismatch["expected"], "startLongTrade")
        self.assertEqual(mismatch["actual"], "changedLongTrade")

    def test_fixture_directory_commands_reject_file_fixture_dir_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "not_a_directory"
            fixture_dir.write_text("not a directory")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "missing.csv",
                                "proves": ["fixture directory validation"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )
            command_args = [
                ["validate-fixtures", "--manifest", str(manifest_path)],
                ["readiness", "--manifest", str(manifest_path)],
                ["readiness-blockers", "--manifest", str(manifest_path)],
                ["audit-fixtures", "--manifest", str(manifest_path)],
                ["pine-input-contract", "--manifest", str(manifest_path)],
                ["pine-output-contract", "--manifest", str(manifest_path)],
                ["export-checklist", "--manifest", str(manifest_path)],
                ["external-runner-pack", "--manifest", str(manifest_path), "--output", str(Path(tmp) / "external-pack")],
                ["prepare-readiness-artifacts", "--manifest", str(manifest_path), "--output", str(Path(tmp) / "artifacts")],
                ["export-pack", "--manifest", str(manifest_path), "--output", str(Path(tmp) / "pack")],
            ]

            for args in command_args:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "lorentzian_classification",
                        *args,
                        "--fixture-dir",
                        str(fixture_dir),
                    ],
                    cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 1, args)
                self.assertIn("fixture path is not a directory", completed.stdout, args)
                self.assertIn(str(fixture_dir), completed.stdout, args)
                self.assertNotIn("Traceback", completed.stdout + completed.stderr, args)

    def test_manifest_commands_reject_directory_manifest_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.mkdir()
            command_args = [
                (["validate-fixtures"], True),
                (["readiness"], True),
                (["readiness-blockers"], True),
                (["audit-fixtures"], True),
                (["pine-input-contract"], True),
                (["pine-output-contract"], True),
                (["export-checklist"], True),
                (["external-runner-pack", "--output", str(Path(tmp) / "external-pack")], True),
                (["prepare-readiness-artifacts", "--output", str(Path(tmp) / "artifacts")], True),
                (["export-pack", "--output", str(Path(tmp) / "pack")], True),
                (["pine-export-helper", "--manifest-case", "missing_case"], False),
            ]

            for args, include_fixture_dir in command_args:
                command = [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    *args,
                    "--manifest",
                    str(manifest_path),
                ]
                if include_fixture_dir:
                    command.extend(["--fixture-dir", str(fixture_dir)])
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 1, args)
                self.assertIn("invalid parity manifest", completed.stdout, args)
                self.assertIn("manifest path is not a file", completed.stdout, args)
                self.assertNotIn("Traceback", completed.stdout + completed.stderr, args)

    def test_manifest_commands_reject_missing_manifest_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "missing_manifest.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(Path(tmp) / "pack"),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid parity manifest", completed.stdout)
        self.assertIn("manifest not found", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_cli_validate_fixtures_runs_known_pine_exports(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "lorentzian_classification",
                "validate-fixtures",
                "--fixture-dir",
                str(self.fixture_dir),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
            capture_output=True,
            text=True,
        )
        self.assertIn(completed.returncode, {0, 1}, completed.stdout + completed.stderr)
        if completed.returncode == 1:
            self.assertTrue(
                "source validation failed:" in completed.stdout
                or "Pine input contract failed:" in completed.stdout
                or "Pine output contract failed:" in completed.stdout,
                completed.stdout,
            )
        for filename, _include_full_history in PARITY_FIXTURES:
            self.assertIn(f"PASS {filename}", completed.stdout)

    def test_cli_require_full_coverage_fails_for_uncovered_manifest_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "dynamic_exits",
                                "filename": "dynamic_exits.csv",
                                "python_smoke_fixture": "dynamic_exits.csv",
                                "proves": ["dynamic exit parity"],
                                "settings": {"use_dynamic_exits": True},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--require-full-coverage",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("coverage incomplete", completed.stdout)
        self.assertIn("dynamic_exits (dynamic_exits.csv)", completed.stdout)
        self.assertIn("missing export action details:", completed.stdout)
        self.assertIn("equivalent CLI flags: --use-dynamic-exits", completed.stdout)
        self.assertIn("Pine export helper command: PYTHONPATH=ports/python python3 -m", completed.stdout)
        self.assertIn("--manifest-case dynamic_exits", completed.stdout)
        self.assertIn("full helper command: PYTHONPATH=ports/python python3 -m", completed.stdout)

    def test_cli_require_full_coverage_treats_required_export_directory_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "known.csv").write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            (fixture_dir / "planned.csv").mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "planned_case",
                                "filename": "planned.csv",
                                "python_smoke_fixture": "known.csv",
                                "proves": ["required export must be a CSV file"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--require-full-coverage",
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["summary"]["required_present"], 0)
        self.assertEqual(payload["summary"]["required_missing"], 1)
        self.assertEqual(payload["summary"]["required_failed"], 0)
        self.assertFalse(payload["required_uncovered_fixtures"][0]["present"])
        self.assertIsNone(payload["required_uncovered_fixtures"][0]["path"])
        self.assertEqual(payload["required_action_items"][0]["status"], "missing")

    def test_cli_require_full_coverage_validates_planned_exports_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "planned.csv"
            write_full_export_fixture(fixture_path, Settings())

            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "planned_case",
                                "filename": "planned.csv",
                                "python_smoke_fixture": "planned.csv",
                                "proves": ["planned export schema validation"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--require-full-coverage",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("PASS planned.csv", completed.stdout)
        self.assertNotIn("coverage incomplete", completed.stdout)

    def test_cli_require_full_coverage_rejects_minimum_only_required_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "planned.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "planned_case",
                                "filename": "planned.csv",
                                "python_smoke_fixture": "planned.csv",
                                "proves": ["full instrumentation is required for final coverage"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--require-full-coverage",
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        record = payload["required_uncovered_fixtures"][0]
        self.assertFalse(record["passed"])
        self.assertIn("Backtest Stream", record["schema"]["missing_required_columns"])
        self.assertIn("Open Long Alert", record["schema"]["missing_full_numeric_columns"])
        self.assertIn("Trade Stats Visible", record["schema"]["missing_full_numeric_columns"])
        self.assertIn(SETTINGS_FINGERPRINT_COLUMN, record["schema"]["missing_required_columns"])

    def test_cli_require_full_coverage_rejects_missing_settings_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "planned.csv"
            write_full_export_fixture(fixture_path, Settings(), include_fingerprint=False)
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "planned_case",
                                "filename": "planned.csv",
                                "python_smoke_fixture": "planned.csv",
                                "proves": ["settings identity is required for final coverage"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--require-full-coverage",
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        record = json.loads(completed.stdout)["required_uncovered_fixtures"][0]
        self.assertFalse(record["passed"])
        self.assertIn(SETTINGS_FINGERPRINT_COLUMN, record["schema"]["missing_required_columns"])
        self.assertEqual(record["schema"]["settings_fingerprint"]["expected"], settings_fingerprint(Settings()))
        self.assertFalse(record["schema"]["settings_fingerprint"]["present"])
        self.assertFalse(record["schema"]["settings_fingerprint"]["valid"])

    def test_cli_require_full_coverage_rejects_wrong_settings_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "planned.csv"
            write_full_export_fixture(fixture_path, Settings(), fingerprint=settings_fingerprint(Settings()) + 1)
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "planned_case",
                                "filename": "planned.csv",
                                "python_smoke_fixture": "planned.csv",
                                "proves": ["settings identity is required for final coverage"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--require-full-coverage",
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        record = json.loads(completed.stdout)["required_uncovered_fixtures"][0]
        fingerprint = record["schema"]["settings_fingerprint"]
        self.assertFalse(record["passed"])
        self.assertEqual(
            record["error"],
            f"settings fingerprint mismatch: expected {settings_fingerprint(Settings())}, mismatched rows 1",
        )
        self.assertEqual(fingerprint["expected"], settings_fingerprint(Settings()))
        self.assertTrue(fingerprint["present"])
        self.assertEqual(fingerprint["mismatch_count"], 1)
        self.assertFalse(fingerprint["valid"])

    def test_cli_validate_fixtures_json_reports_parity_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "known.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "known",
                                "filename": "known.csv",
                                "settings": {},
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_required",
                                "filename": "missing_required.csv",
                                "python_smoke_fixture": "known.csv",
                                "proves": ["missing required export reporting"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--require-full-coverage",
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["summary"]["fixtures_passed"], 1)
        self.assertEqual(payload["summary"]["required_missing"], 1)
        self.assertEqual(payload["fixtures"][0]["summary"]["rows"], 1)
        self.assertEqual(payload["missing_required"], ["missing_required (missing_required.csv)"])
        self.assertEqual(payload["required_action_items"][0]["name"], "missing_required")
        self.assertEqual(payload["required_action_items"][0]["filename"], "missing_required.csv")
        self.assertEqual(payload["required_action_items"][0]["status"], "missing")
        self.assertIn("minimum_export_columns", payload["required_action_items"][0])

    def test_readiness_reports_not_ready_with_missing_required_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "known.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "known",
                                "filename": "known.csv",
                                "settings": {},
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "dynamic_exits",
                                "filename": "dynamic_exits.csv",
                                "python_smoke_fixture": "known.csv",
                                "proves": ["dynamic exit readiness reporting"],
                                "settings": {"use_dynamic_exits": True},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "readiness",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["summary"]["fixtures_passed"], 1)
        self.assertEqual(payload["summary"]["required_missing"], 1)
        self.assertEqual(payload["summary"]["required_settings_smoke_passed"], 1)
        self.assertEqual(payload["required_action_items"][0]["status"], "missing")
        self.assertIn("--use-dynamic-exits", payload["required_action_items"][0]["cli_flags"])
        self.assertIn("pine-export-helper", payload["required_action_items"][0]["pine_export_helper_command"])
        self.assertEqual(payload["required_action_items"][0]["pine_export_helper_command_full"][-1], "--full")
        self.assertIn("export-pack", payload["required_export_workflow"]["export_pack_command"])
        self.assertIn("verify-export-pack", payload["required_export_workflow"]["verify_export_pack_command"])
        self.assertEqual(payload["required_export_workflow"]["output_dir"], "/tmp/lorentzian-export-pack")

    def test_readiness_blockers_reports_remaining_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external With Spaces"
            files = root / "Files"
            indicator_dir = root / "Indicators" / "LorentzianClassification"
            scripts_dir = root / "Scripts"
            files.mkdir(parents=True)
            indicator_dir.mkdir(parents=True)
            scripts_dir.mkdir(parents=True)
            known = files / "known.csv"
            known.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            external_report = files / "external_report.csv"
            external_report.write_text(
                "\n".join(
                    [
                        "time,close,tv_prediction,external_prediction,pred_match,tv_direction,external_direction,dir_match",
                        "2026-01-01,1.15,0,0,1,0,0,1",
                    ]
                )
            )
            indicator = indicator_dir / "LorentzianClassification.external_src"
            indicator.write_text(external_indicator_input_contract_source())
            indicator_artifact = indicator.with_suffix(".compiled")
            indicator_artifact.write_text("compiled\n")
            script = scripts_dir / "LorentzianParityCheck.external_src"
            script.write_text(external_parity_script_contract_source())
            script_artifact = script.with_suffix(".compiled")
            script_artifact.write_text("compiled script\n")
            os.utime(external_report, (2000, 2000))
            os.utime(indicator, (3000, 3000))
            os.utime(indicator_artifact, (4000, 4000))
            os.utime(script, (3500, 3500))
            os.utime(script_artifact, (4500, 4500))
            staged_exports = root / "TradingView Downloads"
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [{"name": "known", "filename": "known.csv", "settings": {}}],
                        "external_sources": [
                            {
                                "name": "LorentzianClassification",
                                "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                                "role": "canonical external",
                            }
                        ],
                        "external_parity_reports": [
                            {
                                "name": "external_report",
                                "filename": "external_report.csv",
                                "role": "unit test report",
                                "input_filename": "known.csv",
                                "include_full_history": False,
                                "required": True,
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "dynamic_exits",
                                "filename": "dynamic_exits.csv",
                                "python_smoke_fixture": "known.csv",
                                "proves": ["readiness blocker action"],
                                "settings": {"use_dynamic_exits": True},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "readiness-blockers",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--pine-export-source-dir",
                    str(staged_exports),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["blocker_count"], 2)
        blockers = {blocker["kind"]: blocker for blocker in payload["blockers"]}
        self.assertEqual(blockers["external_parity_reports"]["count"], 1)
        self.assertEqual(blockers["external_parity_reports"]["stale"], 1)
        self.assertEqual(blockers["external_parity_reports"]["actions"][0]["script_inputs"]["InpInputFile"], "known.csv")
        runner_command = blockers["external_parity_reports"]["workflow"]["external_runner_pack_command"]
        self.assertIn("external-runner-pack", runner_command)
        self.assertIn("--only-failing", runner_command)
        self.assertIn(str(root), runner_command)
        runner_command_string = blockers["external_parity_reports"]["workflow"]["external_runner_pack_command_string"]
        self.assertIn("external-runner-pack", runner_command_string)
        self.assertIn(shlex.quote(str(root)), runner_command_string)
        verify_runner_command = blockers["external_parity_reports"]["workflow"]["verify_external_runner_pack_command"]
        self.assertIn("verify-external-runner-pack", verify_runner_command)
        verify_runner_command_string = blockers["external_parity_reports"]["workflow"]["verify_external_runner_pack_command_string"]
        self.assertIn("verify-external-runner-pack", verify_runner_command_string)
        self.assertEqual(blockers["required_pine_exports"]["count"], 1)
        self.assertEqual(blockers["required_pine_exports"]["actions"][0]["name"], "dynamic_exits")
        self.assertIn("export-pack", blockers["required_pine_exports"]["workflow"]["export_pack_command"])
        export_command_string = blockers["required_pine_exports"]["workflow"]["export_pack_command_string"]
        self.assertIn("export-pack", export_command_string)
        self.assertIn(shlex.quote(str(root)), export_command_string)
        import_command = blockers["required_pine_exports"]["workflow"]["import_pine_exports_command"]
        self.assertIn("import-pine-exports", import_command)
        self.assertIn(str(staged_exports), import_command)
        import_command_string = blockers["required_pine_exports"]["workflow"]["import_pine_exports_command_string"]
        self.assertIn(f"import-pine-exports {shlex.quote(str(staged_exports))}", import_command_string)
        self.assertIn(shlex.quote(str(root)), import_command_string)
        readiness_command = blockers["required_pine_exports"]["workflow"]["readiness_command"]
        self.assertIn("readiness", readiness_command)
        readiness_command_string = blockers["required_pine_exports"]["workflow"]["readiness_command_string"]
        self.assertIn("readiness --fixture-dir", readiness_command_string)
        readiness_workflow = payload["readiness_artifacts_workflow"]
        self.assertEqual(readiness_workflow["output_dir"], "/tmp/lorentzian-readiness-artifacts")
        self.assertIn("prepare-readiness-artifacts", readiness_workflow["prepare_readiness_artifacts_command"])
        self.assertIn("--clean-stale", readiness_workflow["prepare_readiness_artifacts_command"])
        self.assertIn("verify-readiness-artifacts", readiness_workflow["verify_readiness_artifacts_command"])
        self.assertIn("prepare-readiness-artifacts", readiness_workflow["prepare_readiness_artifacts_command_string"])
        self.assertIn("verify-readiness-artifacts", readiness_workflow["verify_readiness_artifacts_command_string"])
        self.assertIn(shlex.quote(str(root)), readiness_workflow["prepare_readiness_artifacts_command_string"])

    def test_validate_fixtures_and_readiness_share_required_export_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "known.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "known",
                                "filename": "known.csv",
                                "settings": {},
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_required",
                                "filename": "missing_required.csv",
                                "python_smoke_fixture": "known.csv",
                                "proves": ["required settings smoke reporting"],
                                "settings": {"use_worst_case": True},
                            }
                        ],
                    }
                )
            )

            env = {**os.environ, "PYTHONPATH": str(PYTHON_PORT)}
            validation = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--require-full-coverage",
                    "--json",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            readiness = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "readiness",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertEqual(validation.returncode, 1)
        self.assertEqual(readiness.returncode, 1)
        validation_payload = json.loads(validation.stdout)
        readiness_payload = json.loads(readiness.stdout)
        self.assertFalse(validation_payload["passed"])
        self.assertFalse(readiness_payload["ready"])
        for key in [
            "fixtures",
            "required_uncovered_fixtures",
            "required_settings_smoke",
            "missing_required",
            "required_action_items",
            "source_errors",
            "summary",
        ]:
            self.assertEqual(validation_payload[key], readiness_payload[key])

    def test_readiness_reports_ready_when_required_exports_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "planned.csv"
            write_full_export_fixture(fixture_path, Settings())
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "planned",
                                "filename": "planned.csv",
                                "python_smoke_fixture": "planned.csv",
                                "proves": ["readiness required export handling"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "readiness",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("readiness: READY", completed.stdout)
        self.assertIn("required_passed=1/1", completed.stdout)
        self.assertIn("required_settings_smoke_passed=1/1", completed.stdout)

    def test_validate_fixtures_rejects_incomplete_pine_export_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "incomplete.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,0,0,,,,",
                    ]
                )
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "incomplete",
                                "filename": "incomplete.csv",
                                "settings": {},
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        record = payload["fixtures"][0]
        self.assertFalse(record["passed"])
        self.assertIsNone(record["summary"])
        self.assertIn("F1_RSI", record["schema"]["missing_required_columns"])
        self.assertIn("missing required export columns", record["error"])

    def test_parity_command_rejects_incomplete_pine_export_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "incomplete.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,0,0,,,,",
                    ]
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "parity",
                    str(fixture_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid Pine export schema", completed.stdout)
        self.assertIn("F1_RSI", completed.stdout)

    def test_parity_command_rejects_duplicate_pine_export_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "duplicate_columns.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,0,1,0,0,0,0,1.12,0,0,,,,",
                    ]
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "parity",
                    str(fixture_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid Pine export schema", completed.stdout)
        self.assertIn("duplicate export columns:", completed.stdout)
        self.assertIn("F1_RSI", completed.stdout)

    def test_parity_command_rejects_missing_input_file_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "missing.csv"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "parity",
                    str(fixture_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("input file not found", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_parity_command_rejects_directory_input_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "input_dir"
            fixture_path.mkdir()

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "parity",
                    str(fixture_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("input path is not a file", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_parity_command_rejects_missing_output_directory_without_traceback(self) -> None:
        fixture_path = self.fixture_dir / "pine_coinbase_btcusd_1d_limited_history.csv"
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "missing" / "mismatches.csv"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "parity",
                    str(fixture_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            output_exists = output_path.exists()

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid output path", completed.stdout)
        self.assertIn("output directory not found", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)
        self.assertFalse(output_exists)

    def test_validate_fixtures_rejects_file_output_mismatch_directory_without_traceback(self) -> None:
        source_path = self.fixture_dir / "pine_coinbase_btcusd_1d_limited_history.csv"
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "tampered.csv"
            with source_path.open(newline="") as source_handle:
                reader = csv.DictReader(source_handle)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
            self.assertTrue(rows, f"fixture has no rows: {source_path}")
            rows[-1]["Prediction"] = "999999"
            with fixture_path.open("w", newline="") as fixture_handle:
                writer = csv.DictWriter(fixture_handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "tampered",
                                "filename": "tampered.csv",
                                "settings": {"include_full_history": False},
                            }
                        ]
                    }
                )
            )
            output_mismatches = Path(tmp) / "not_a_directory"
            output_mismatches.write_text("existing file")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output-mismatches",
                    str(output_mismatches),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("FAIL tampered.csv", completed.stdout)
        self.assertIn("invalid output mismatch directory", completed.stdout)
        self.assertIn("output directory path is not a directory", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_validate_fixtures_rejects_duplicate_pine_export_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "duplicate_columns.csv").write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,0,1,0,0,0,0,1.12,0,0,,,,",
                    ]
                )
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "duplicate_columns",
                                "filename": "duplicate_columns.csv",
                                "settings": {},
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        record = payload["fixtures"][0]
        self.assertFalse(record["passed"])
        self.assertEqual(record["schema"]["duplicate_columns"], ["F1_RSI"])
        self.assertIn("duplicate export columns: F1_RSI", record["error"])

    def test_pine_export_schema_rejects_header_only_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "header_only.csv"
            fixture_path.write_text(
                "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell\n"
            )

            schema = pine_export_schema(fixture_path)

        self.assertFalse(schema["valid"])
        self.assertEqual(schema["row_count"], 0)
        self.assertFalse(schema["missing_required_columns"])

    def test_parity_command_rejects_header_only_pine_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "header_only.csv"
            fixture_path.write_text(
                "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell\n"
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "parity",
                    str(fixture_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid Pine export schema", completed.stdout)
        self.assertIn("no data rows", completed.stdout)

    def test_validate_fixtures_rejects_header_only_pine_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "header_only.csv").write_text(
                "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell\n"
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "header_only",
                                "filename": "header_only.csv",
                                "settings": {},
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        record = payload["fixtures"][0]
        self.assertFalse(record["passed"])
        self.assertEqual(record["schema"]["row_count"], 0)
        self.assertEqual(record["error"], "no data rows")

    def test_parity_command_rejects_malformed_pine_export_rows_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "malformed.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,not-a-price,1.20,1.00,1.15,0,0,0,0,0,1.12,0,0,,,,",
                    ]
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "parity",
                    str(fixture_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid Pine export data", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_validate_fixtures_rejects_malformed_pine_export_rows_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "malformed.csv").write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,not-a-price,1.20,1.00,1.15,0,0,0,0,0,1.12,0,0,,,,",
                    ]
                )
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "malformed",
                                "filename": "malformed.csv",
                                "settings": {},
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        record = payload["fixtures"][0]
        self.assertFalse(record["passed"])
        self.assertIn("invalid Pine export data", record["error"])

    def test_pine_export_schema_accepts_slot_feature_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "custom_slots.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_CCI,F2_RSI,F3_WT,F4_ADX,F5_CCI,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,0,0,0,0,0,1.12,0,0,,,,",
                    ]
                )
            )

            schema = pine_export_schema(fixture_path)

        self.assertTrue(schema["valid"])
        self.assertFalse(schema["missing_required_columns"])
        self.assertFalse(schema["duplicate_columns"])

    def test_pine_export_schema_rejects_duplicate_export_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "duplicate_columns.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,0,1,0,0,0,0,1.12,0,0,,,,",
                    ]
                )
            )

            schema = pine_export_schema(fixture_path)

        self.assertFalse(schema["valid"])
        self.assertEqual(schema["duplicate_columns"], ["F1_RSI"])
        self.assertFalse(schema["missing_required_columns"])

    def test_pine_export_schema_requires_manifest_feature_columns_when_settings_are_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "wrong_custom_feature.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,0,0,0,0,0,1.12,0,0,,,,",
                    ]
                )
            )
            settings = settings_from_mapping({"f1": "CCI:34:2"})

            schema = pine_export_schema(fixture_path, settings)

        self.assertFalse(schema["valid"])
        self.assertIn("F1_CCI", schema["missing_required_columns"])
        self.assertEqual(schema["expected_feature_columns"]["F1"], "F1_CCI")

    def test_reader_uses_settings_feature_columns_when_default_columns_are_also_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "ambiguous_custom_feature.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F1_CCI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,1,2,0,0,0,0,1.12,0,0,,,,",
                    ]
                )
            )
            settings = settings_from_mapping({"f1": "CCI:34:2"})

            rows, _price_scale = read_tradingview_csv(fixture_path, feature_columns=feature_export_columns(settings))

        self.assertEqual(rows[0].f1, 2)

    def test_export_checklist_json_reports_required_pine_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "present_case",
                                "filename": "present.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["present export handling"],
                                "settings": {"include_full_history": True},
                            },
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["missing export reporting"],
                                "settings": {
                                    "use_dynamic_exits": True,
                                    "feature_count": 3,
                                    "f1": "CCI:34:2",
                                },
                            },
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-checklist",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["summary"], {"missing": 1, "present": 1, "total": 2})
        self.assertIn("export-pack", payload["export_workflow"]["export_pack_command"])
        self.assertIn("verify-export-pack", payload["export_workflow"]["verify_export_pack_command"])
        self.assertEqual(payload["export_workflow"]["output_dir"], "/tmp/lorentzian-export-pack")
        self.assertTrue(payload["exports"][0]["present"])
        self.assertFalse(payload["exports"][1]["present"])
        self.assertEqual(payload["exports"][1]["settings"]["feature_count"], 3)
        expected_settings = settings_from_mapping(payload["exports"][1]["settings"])
        self.assertEqual(
            payload["exports"][1]["minimum_export_columns"],
            pine_export_columns_for_settings(expected_settings, include_full=False),
        )
        self.assertEqual(
            payload["exports"][1]["full_instrumented_export_columns"],
            pine_export_columns_for_settings(expected_settings, include_full=True) + [SETTINGS_FINGERPRINT_COLUMN],
        )
        self.assertEqual(payload["exports"][1]["settings_fingerprint"], settings_fingerprint(expected_settings))
        self.assertEqual(
            payload["exports"][1]["pine_export_series"],
            pine_export_series_for_settings(expected_settings, include_full=True),
        )
        self.assertIn("F1_CCI", payload["exports"][1]["minimum_export_columns"])
        self.assertIn("F2_WT", payload["exports"][1]["minimum_export_columns"])
        export_series = {row["column"]: row for row in payload["exports"][1]["pine_export_series"]}
        self.assertEqual(export_series["F1_CCI"]["pine_expression"], "featureSeries.f1")
        self.assertEqual(export_series["Prediction"]["pine_expression"], "prediction")
        self.assertEqual(export_series["Buy"]["pine_expression"], "startLongTrade ? 1 : na")
        self.assertEqual(export_series["Backtest Stream"]["pine_expression"], "backTestStream")
        self.assertEqual(export_series["Trade Stats Visible"]["pine_expression"], "showTradeStats ? 1 : 0")
        self.assertEqual(export_series["Table WL Ratio"]["pine_expression"], "totalWins / totalLosses")
        self.assertEqual(export_series["Kernel Plot Color"]["export_mode"], "encoded_plot_data_window")
        self.assertIn("120", export_series["Kernel Plot Color"]["pine_expression"])
        titles = [row["title"] for row in payload["exports"][1]["tradingview_settings"]]
        self.assertIn("Use Dynamic Exits", titles)
        self.assertIn("Feature Count", titles)
        self.assertIn("Feature 1", titles)
        self.assertIn("Parameter A", titles)
        self.assertIn("Parameter B", titles)
        self.assertIn("--use-dynamic-exits", payload["exports"][1]["cli_flags"])
        self.assertIn("--feature-count", payload["exports"][1]["cli_flags"])
        self.assertIn("CCI:34:2", payload["exports"][1]["cli_flags"])
        self.assertEqual(
            payload["exports"][1]["pine_export_helper_command"],
            [
                "PYTHONPATH=ports/python",
                "python3",
                "-m",
                "lorentzian_classification",
                "pine-export-helper",
                "--manifest",
                str(manifest_path),
                "--manifest-case",
                "missing_case",
            ],
        )
        self.assertEqual(payload["exports"][1]["pine_export_helper_command_full"][-1], "--full")

    def test_export_checklist_text_includes_tradingview_export_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "kernel_smoothing",
                                "filename": "kernel_smoothing.csv",
                                "python_smoke_fixture": "kernel_smoothing.csv",
                                "proves": ["kernel crossover alert parity"],
                                "settings": {
                                    "include_full_history": True,
                                    "use_kernel_smoothing": True,
                                    "kernel_lag": 1,
                                },
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-checklist",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("TradingView inputs:", completed.stdout)
        self.assertIn("Include Full History [General Settings] (includeFullHistory): True", completed.stdout)
        self.assertIn("Enhance Kernel Smoothing [Kernel Settings] (useKernelSmoothing) inline=1: True", completed.stdout)
        self.assertIn("Lag [Kernel Settings] (lag) inline=1: 1", completed.stdout)
        self.assertIn("Pine export helper command: PYTHONPATH=ports/python python3 -m", completed.stdout)
        self.assertIn("--manifest-case kernel_smoothing", completed.stdout)
        self.assertIn("export pack command: PYTHONPATH=ports/python python3 -m", completed.stdout)
        self.assertIn("verify export pack command: PYTHONPATH=ports/python python3 -m", completed.stdout)
        self.assertIn("full helper command: PYTHONPATH=ports/python python3 -m", completed.stdout)
        self.assertIn("--full", completed.stdout)
        self.assertIn("minimum export columns: time, open, high, low, close", completed.stdout)
        self.assertIn("full instrumented export columns: time, open, high, low, close", completed.stdout)
        self.assertIn(SETTINGS_FINGERPRINT_COLUMN, completed.stdout)
        self.assertIn("settings fingerprint:", completed.stdout)
        self.assertIn("Pine export series:", completed.stdout)
        self.assertIn("Prediction [plot_data_window]: prediction", completed.stdout)
        self.assertIn("Buy [plot_data_window]: startLongTrade ? 1 : na", completed.stdout)
        self.assertIn("Kernel Plot Color [encoded_plot_data_window]: showKernelEstimate", completed.stdout)

    def test_export_checklist_can_filter_to_one_manifest_case(self) -> None:
        manifest_path = ROOT / "tests" / "parity" / "fixtures_manifest.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "lorentzian_classification",
                "export-checklist",
                "--fixture-dir",
                str(self.fixture_dir),
                "--manifest",
                str(manifest_path),
                "--case",
                "coinbase_daily_custom_features_count3",
                "--json",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["summary"], {"missing": 1, "present": 0, "total": 1})
        self.assertEqual(len(payload["exports"]), 1)
        self.assertEqual(payload["exports"][0]["name"], "coinbase_daily_custom_features_count3")
        self.assertIn("F1_CCI", payload["exports"][0]["minimum_export_columns"])
        self.assertIn("--manifest-case", payload["exports"][0]["pine_export_helper_command"])

    def test_export_checklist_workflow_prefers_external_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            external_root = Path(tmp) / "external"
            fixture_dir = external_root / "Files"
            fixture_dir.mkdir(parents=True)
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "missing.csv",
                                "proves": ["workspace-root workflow command"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-checklist",
                    "--fixture-dir",
                    str(external_root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        command = payload["export_workflow"]["export_pack_command"]
        fixture_arg_index = command.index("--fixture-dir") + 1
        self.assertEqual(command[fixture_arg_index], str(external_root))
        self.assertNotEqual(command[fixture_arg_index], str(fixture_dir))

    def test_export_checklist_rejects_unknown_case_filter(self) -> None:
        manifest_path = ROOT / "tests" / "parity" / "fixtures_manifest.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "lorentzian_classification",
                "export-checklist",
                "--fixture-dir",
                str(self.fixture_dir),
                "--manifest",
                str(manifest_path),
                "--case",
                "not_a_manifest_case",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("export checklist case not found: not_a_manifest_case", completed.stdout)

    def test_export_pack_writes_missing_full_helper_snippets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "present_case",
                                "filename": "present.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["present export handling"],
                                "settings": {},
                            },
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["missing export helper generation"],
                                "settings": {
                                    "use_dynamic_exits": True,
                                    "feature_count": 3,
                                    "f1": "CCI:34:2",
                                },
                            },
                        ],
                    }
                )
            )
            output_dir = Path(tmp) / "pack"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["summary"], {"missing": 1, "present": 0, "total": 1})
            self.assertEqual(payload["exports"][0]["name"], "missing_case")
            snippet_path = Path(payload["exports"][0]["helper_snippet"])
            self.assertTrue(snippet_path.exists())
            self.assertFalse((output_dir / "present_case.pine").exists())
            snippet = snippet_path.read_text()
            self.assertTrue(snippet.startswith(EXPORT_PACK_HEADER))
            self.assertIn("// Manifest case: missing_case", snippet)
            self.assertIn("// Target CSV filename: missing.csv", snippet)
            self.assertIn('plot(featureSeries.f1, "F1_CCI", display=display.data_window)', snippet)
            self.assertIn('plot(backTestStream, "Backtest Stream", display=display.data_window)', snippet)
            self.assertIn('plot(showKernelEstimate ?', snippet)
            self.assertIn('"Kernel Plot Color", display=display.data_window)', snippet)
            missing_case_settings = settings_from_mapping(
                {"use_dynamic_exits": True, "feature_count": 3, "f1": "CCI:34:2"}
            )
            self.assertIn(
                f"// Settings Fingerprint: {settings_fingerprint(missing_case_settings)} "
                f"({SETTINGS_FINGERPRINT_COLUMN})",
                snippet,
            )
            self.assertIn(
                f'plot({settings_fingerprint(missing_case_settings)}, '
                '"Settings Fingerprint", display=display.data_window)',
                snippet,
            )
            index = json.loads((output_dir / "export_pack.json").read_text())
            self.assertEqual(index["exports"][0]["helper_snippet"], str(snippet_path))
            self.assertEqual(index["exports"][0]["helper_snippet_sha256"], file_sha256(snippet_path))
            self.assertEqual(index["exports"][0]["pine_source_lock_fingerprint"], index["pine_source_lock_fingerprint"])
            self.assertEqual(len(index["pine_source_lock_fingerprint"]), 64)
            self.assertIn(
                f"// Pine Source Lock Fingerprint: {index['pine_source_lock_fingerprint']}",
                snippet,
            )
            acceptance_manifest_path = Path(index["acceptance_manifest"])
            self.assertTrue(acceptance_manifest_path.exists())
            with acceptance_manifest_path.open(newline="") as handle:
                acceptance_rows = list(csv.DictReader(handle))
            self.assertEqual(len(acceptance_rows), 1)
            self.assertEqual(acceptance_rows[0]["name"], "missing_case")
            self.assertEqual(acceptance_rows[0]["target_csv"], "missing.csv")
            self.assertEqual(acceptance_rows[0]["helper_snippet"], "missing_case.pine")
            self.assertEqual(acceptance_rows[0]["helper_snippet_sha256"], file_sha256(snippet_path))
            self.assertEqual(
                acceptance_rows[0]["pine_source_lock_fingerprint"],
                index["pine_source_lock_fingerprint"],
            )
            self.assertEqual(
                acceptance_rows[0]["settings_fingerprint"],
                str(settings_fingerprint(missing_case_settings)),
            )
            self.assertEqual(acceptance_rows[0]["settings_fingerprint_column"], SETTINGS_FINGERPRINT_COLUMN)
            self.assertEqual(acceptance_rows[0]["required_full_export_column_count"], "41")
            self.assertIn(SETTINGS_FINGERPRINT_COLUMN, acceptance_rows[0]["required_full_export_columns"])
            self.assertIn("--use-dynamic-exits", acceptance_rows[0]["cli_flags"])
            self.assertIn('"feature_count": 3', acceptance_rows[0]["settings_json"])
            readme = (output_dir / "README.md").read_text()
            self.assertIn("Acceptance manifest: `acceptance_manifest.csv`", readme)
            self.assertIn(
                f"Pine source lock fingerprint: `{index['pine_source_lock_fingerprint']}`",
                readme,
            )
            self.assertIn("Target CSV: `missing.csv`", readme)
            self.assertIn("Helper snippet: `missing_case.pine`", readme)
            self.assertIn(
                f"Settings fingerprint: `{settings_fingerprint(missing_case_settings)}` "
                f"in `{SETTINGS_FINGERPRINT_COLUMN}`",
                readme,
            )
            self.assertIn(
                f"Pine source lock fingerprint: `{index['pine_source_lock_fingerprint']}`",
                readme,
            )
            self.assertIn("Required full export columns: `41`", readme)
            self.assertIn("- TradingView settings:", readme)
            self.assertIn(
                "General Settings / Use Dynamic Exits: `True` "
                "(setting `use_dynamic_exits`, inline `exits`)",
                readme,
            )
            self.assertIn(
                "Feature Engineering / Feature 1: `CCI` "
                "(setting `f1`, variable `f1_string`, inline `01`)",
                readme,
            )
            self.assertIn(
                "Feature Engineering / Parameter A: `34` "
                "(setting `f1`, variable `f1_paramA`, inline `02`)",
                readme,
            )

    def test_import_pine_exports_validates_and_copies_staged_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "downloads"
            fixture_dir = Path(tmp) / "fixtures"
            source_dir.mkdir()
            fixture_dir.mkdir()
            settings = Settings(show_exits=True)
            write_full_export_fixture(source_dir / "missing.csv", settings)
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "smoke.csv",
                                "proves": ["validated import of downloaded Pine export"],
                                "settings": {"show_exits": True},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "import-pine-exports",
                    str(source_dir),
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            copied_path_exists = (fixture_dir / "missing.csv").exists()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["summary"]["copied"], 1)
        self.assertEqual(payload["imports"][0]["status"], "copied")
        self.assertTrue(copied_path_exists)

    def test_import_pine_exports_searches_multiple_source_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_source_dir = Path(tmp) / "empty-downloads"
            source_dir = Path(tmp) / "desktop-downloads"
            fixture_dir = Path(tmp) / "fixtures"
            empty_source_dir.mkdir()
            source_dir.mkdir()
            fixture_dir.mkdir()
            settings = Settings(show_exits=True)
            write_full_export_fixture(source_dir / "missing.csv", settings)
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "smoke.csv",
                                "proves": ["validated import from multiple staging directories"],
                                "settings": {"show_exits": True},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "import-pine-exports",
                    str(empty_source_dir),
                    str(source_dir),
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            copied_path_exists = (fixture_dir / "missing.csv").exists()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["source_dirs"], [str(empty_source_dir), str(source_dir)])
        self.assertEqual(payload["summary"]["copied"], 1)
        self.assertEqual(payload["imports"][0]["source"], str(source_dir / "missing.csv"))
        self.assertEqual(
            payload["imports"][0]["candidate_sources"],
            [str(empty_source_dir / "missing.csv"), str(source_dir / "missing.csv")],
        )
        self.assertTrue(copied_path_exists)

    def test_import_pine_exports_rejects_settings_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "downloads"
            fixture_dir = Path(tmp) / "fixtures"
            source_dir.mkdir()
            fixture_dir.mkdir()
            write_full_export_fixture(source_dir / "missing.csv", Settings(show_exits=True), fingerprint=123)
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "smoke.csv",
                                "proves": ["reject wrong TradingView settings export"],
                                "settings": {"show_exits": True},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "import-pine-exports",
                    str(source_dir),
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            copied_path_exists = (fixture_dir / "missing.csv").exists()

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["valid"])
        self.assertEqual(payload["summary"]["failed"], 1)
        self.assertEqual(payload["imports"][0]["status"], "invalid_source")
        self.assertIn("settings fingerprint mismatch", payload["imports"][0]["error"])
        self.assertFalse(copied_path_exists)

    def test_export_pack_rejects_ambiguous_helper_snippet_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "case one",
                                "filename": "case_one.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["helper filename collision"],
                                "settings": {},
                            },
                            {
                                "name": "case/one",
                                "filename": "case_slash_one.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["helper filename collision"],
                                "settings": {},
                            },
                        ],
                    }
                )
            )
            output_dir = Path(tmp) / "pack"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("export pack helper snippet filenames are ambiguous", completed.stdout)
        self.assertIn("case_one.pine: case one, case/one", completed.stdout)
        self.assertFalse(output_dir.exists())
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_export_pack_rejects_case_insensitive_helper_snippet_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "Case",
                                "filename": "case_upper.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["case insensitive helper filename collision"],
                                "settings": {},
                            },
                            {
                                "name": "case",
                                "filename": "case_lower.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["case insensitive helper filename collision"],
                                "settings": {},
                            },
                        ],
                    }
                )
            )
            output_dir = Path(tmp) / "pack"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("export pack helper snippet filenames are ambiguous", completed.stdout)
        self.assertIn("Case.pine: Case, case", completed.stdout)
        self.assertFalse(output_dir.exists())
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_export_pack_rejects_file_output_directory_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["export pack output directory validation"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )
            output_dir = Path(tmp) / "pack"
            output_dir.write_text("existing file")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid export pack output directory", completed.stdout)
        self.assertIn("output directory path is not a directory", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_export_pack_rejects_file_output_directory_parent_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["export pack output directory validation"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )
            output_parent = Path(tmp) / "not_a_parent"
            output_parent.write_text("existing file")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_parent / "pack"),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid export pack output directory", completed.stdout)
        self.assertIn("output directory parent is not a directory", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_verify_export_pack_rejects_file_input_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "pack"
            input_path.write_text("not a directory")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(input_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("export pack input is not a directory", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_verify_export_pack_rejects_directory_index_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pack_dir = Path(tmp) / "pack"
            pack_dir.mkdir()
            (pack_dir / "export_pack.json").mkdir()

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(pack_dir),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid export pack index", completed.stdout)
        self.assertIn("path is not a file", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_verify_export_pack_rejects_directory_artifacts_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["export pack verification path validation"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            def generate_pack(output_dir: Path) -> dict[str, object]:
                generated = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "lorentzian_classification",
                        "export-pack",
                        "--fixture-dir",
                        str(fixture_dir),
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(output_dir),
                        "--json",
                    ],
                    cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
                return json.loads(generated.stdout)

            artifact_cases = []
            acceptance_pack = Path(tmp) / "acceptance_pack"
            acceptance_payload = generate_pack(acceptance_pack)
            acceptance_path = Path(str(acceptance_payload["acceptance_manifest"]))
            acceptance_path.unlink()
            acceptance_path.mkdir()
            artifact_cases.append(
                (acceptance_pack, "acceptance manifest path is not a file")
            )

            helper_pack = Path(tmp) / "helper_pack"
            helper_payload = generate_pack(helper_pack)
            helper_path = Path(str(helper_payload["exports"][0]["helper_snippet"]))
            helper_path.unlink()
            helper_path.mkdir()
            artifact_cases.append((helper_pack, "helper snippet path is not a file"))

            readme_pack = Path(tmp) / "readme_pack"
            generate_pack(readme_pack)
            readme_path = readme_pack / "README.md"
            readme_path.unlink()
            readme_path.mkdir()
            artifact_cases.append((readme_pack, "export pack README path is not a file"))

            for pack_dir, expected_error in artifact_cases:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "lorentzian_classification",
                        "verify-export-pack",
                        str(pack_dir),
                        "--json",
                    ],
                    cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 1)
                payload = json.loads(completed.stdout)
                self.assertFalse(payload["valid"])
                self.assertIn(expected_error, "\n".join(payload["errors"]))
                self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_verify_export_pack_rejects_windows_absolute_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "sha256": file_sha256(REPO_PINE_LCV6),
                                "allow_debug_markers": False,
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["export pack Windows path validation"],
                                "settings": {"show_exits": True},
                            }
                        ],
                    }
                )
            )
            pine_source_path = fixture_dir / "PineScript" / "Indicators" / "lcv6.pine"
            pine_source_path.parent.mkdir(parents=True)
            pine_source_path.write_text(REPO_PINE_LCV6.read_text())
            output_dir = Path(tmp) / "pack"
            generated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            index_path = output_dir / "export_pack.json"

            index_payload = json.loads(generated.stdout)
            index_payload["acceptance_manifest"] = "C:\\exports\\acceptance_manifest.csv"
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
            rejected_acceptance_path = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(rejected_acceptance_path.returncode, 1)
            acceptance_payload = json.loads(rejected_acceptance_path.stdout)
            self.assertFalse(acceptance_payload["valid"])
            self.assertIn(
                "acceptance manifest path must not be a Windows absolute path",
                "\n".join(acceptance_payload["errors"]),
            )
            self.assertNotIn("Traceback", rejected_acceptance_path.stdout + rejected_acceptance_path.stderr)

            index_payload = json.loads(generated.stdout)
            index_payload["exports"][0]["helper_snippet"] = "D:\\exports\\missing_case.pine"
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
            rejected_helper_path = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(rejected_helper_path.returncode, 1)
            helper_payload = json.loads(rejected_helper_path.stdout)
            self.assertFalse(helper_payload["valid"])
            self.assertIn(
                "missing_case: helper snippet path must not be a Windows absolute path",
                "\n".join(helper_payload["errors"]),
            )
            self.assertNotIn("Traceback", rejected_helper_path.stdout + rejected_helper_path.stderr)

    def test_verify_export_pack_rejects_unsafe_target_csv_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "sha256": file_sha256(REPO_PINE_LCV6),
                                "allow_debug_markers": False,
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["export pack target CSV validation"],
                                "settings": {"show_exits": True},
                            }
                        ],
                    }
                )
            )
            pine_source_path = fixture_dir / "PineScript" / "Indicators" / "lcv6.pine"
            pine_source_path.parent.mkdir(parents=True)
            pine_source_path.write_text(REPO_PINE_LCV6.read_text())
            output_dir = Path(tmp) / "pack"
            generated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            index_path = output_dir / "export_pack.json"
            acceptance_path = output_dir / "acceptance_manifest.csv"
            original_acceptance = acceptance_path.read_text()

            with acceptance_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["target_csv"] = "../missing.csv"
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            rejected_acceptance_target = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(rejected_acceptance_target.returncode, 1)
            acceptance_payload = json.loads(rejected_acceptance_target.stdout)
            self.assertFalse(acceptance_payload["valid"])
            self.assertIn(
                "missing_case: acceptance target_csv must not contain '..'",
                "\n".join(acceptance_payload["errors"]),
            )
            self.assertNotIn("Traceback", rejected_acceptance_target.stdout + rejected_acceptance_target.stderr)

            acceptance_path.write_text(original_acceptance)
            index_payload = json.loads(generated.stdout)
            index_payload["exports"][0]["filename"] = "C:\\exports\\missing.csv"
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
            rejected_export_target = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(rejected_export_target.returncode, 1)
            export_payload = json.loads(rejected_export_target.stdout)
            self.assertFalse(export_payload["valid"])
            self.assertIn(
                "missing_case: target_csv must be a relative path",
                "\n".join(export_payload["errors"]),
            )
            self.assertNotIn("Traceback", rejected_export_target.stdout + rejected_export_target.stderr)

    def test_verify_export_pack_rejects_unsafe_acceptance_helper_snippet_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "sha256": file_sha256(REPO_PINE_LCV6),
                                "allow_debug_markers": False,
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["export pack helper filename validation"],
                                "settings": {"show_exits": True},
                            }
                        ],
                    }
                )
            )
            pine_source_path = fixture_dir / "PineScript" / "Indicators" / "lcv6.pine"
            pine_source_path.parent.mkdir(parents=True)
            pine_source_path.write_text(REPO_PINE_LCV6.read_text())
            output_dir = Path(tmp) / "pack"
            generated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            acceptance_path = output_dir / "acceptance_manifest.csv"
            with acceptance_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["helper_snippet"] = "../missing_case.pine"
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            rejected_helper_snippet = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_helper_snippet.returncode, 1)
            helper_payload = json.loads(rejected_helper_snippet.stdout)
            self.assertFalse(helper_payload["valid"])
            self.assertIn(
                "missing_case: acceptance helper_snippet must not contain '..'",
                "\n".join(helper_payload["errors"]),
            )
            self.assertNotIn("Traceback", rejected_helper_snippet.stdout + rejected_helper_snippet.stderr)

    def test_verify_export_pack_rejects_unsafe_pine_source_lock_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "sha256": file_sha256(REPO_PINE_LCV6),
                                "allow_debug_markers": False,
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["export pack source lock validation"],
                                "settings": {"show_exits": True},
                            }
                        ],
                    }
                )
            )
            pine_source_path = fixture_dir / "PineScript" / "Indicators" / "lcv6.pine"
            pine_source_path.parent.mkdir(parents=True)
            pine_source_path.write_text(REPO_PINE_LCV6.read_text())
            output_dir = Path(tmp) / "pack"
            generated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            index_path = output_dir / "export_pack.json"
            index_payload = json.loads(generated.stdout)
            index_payload["pine_source_locks"][0]["path"] = "../PineScript/Indicators/lcv6.pine"
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

            rejected_source_lock_path = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_source_lock_path.returncode, 1)
            source_lock_payload = json.loads(rejected_source_lock_path.stdout)
            self.assertFalse(source_lock_payload["valid"])
            source_lock_errors = "\n".join(source_lock_payload["errors"])
            self.assertIn("pine_source_locks 1: path must not contain '..'", source_lock_errors)
            self.assertIn("invalid pine_source_locks", source_lock_errors)
            self.assertNotIn("Traceback", rejected_source_lock_path.stdout + rejected_source_lock_path.stderr)

    def test_verify_export_pack_accepts_intact_pack_and_rejects_tampered_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "sha256": file_sha256(REPO_PINE_LCV6),
                                "allow_debug_markers": False,
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["export pack verification"],
                                "settings": {"show_exits": True},
                            }
                        ],
                    }
                )
            )
            pine_source_path = fixture_dir / "PineScript" / "Indicators" / "lcv6.pine"
            pine_source_path.parent.mkdir(parents=True)
            pine_source_path.write_text(REPO_PINE_LCV6.read_text())
            output_dir = Path(tmp) / "pack"
            generated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)

            verified = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
            verification_payload = json.loads(verified.stdout)
            self.assertTrue(verification_payload["valid"])
            self.assertEqual(verification_payload["export_count"], 1)
            index_path = output_dir / "export_pack.json"
            index_payload = json.loads(index_path.read_text())
            outside_acceptance_path = Path(tmp) / "outside_acceptance_manifest.csv"
            outside_acceptance_path.write_text((output_dir / "acceptance_manifest.csv").read_text())
            index_payload["acceptance_manifest"] = str(outside_acceptance_path)
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

            rejected_acceptance_path = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_acceptance_path.returncode, 1)
            rejected_acceptance_path_payload = json.loads(rejected_acceptance_path.stdout)
            self.assertFalse(rejected_acceptance_path_payload["valid"])
            self.assertIn(
                "acceptance manifest path escapes export pack",
                "\n".join(rejected_acceptance_path_payload["errors"]),
            )
            index_payload = json.loads(generated.stdout)
            helper_path = Path(index_payload["exports"][0]["helper_snippet"])
            outside_helper_path = Path(tmp) / "outside_helper.pine"
            outside_helper_path.write_text(helper_path.read_text())
            index_payload["exports"][0]["helper_snippet"] = str(outside_helper_path)
            index_payload["exports"][0]["helper_snippet_sha256"] = file_sha256(outside_helper_path)
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

            rejected_helper_path = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_helper_path.returncode, 1)
            rejected_helper_path_payload = json.loads(rejected_helper_path.stdout)
            self.assertFalse(rejected_helper_path_payload["valid"])
            self.assertIn(
                "helper snippet path escapes export pack",
                "\n".join(rejected_helper_path_payload["errors"]),
            )
            index_payload = json.loads(generated.stdout)
            index_payload["pine_source_lock_fingerprint"] = "g" * 64
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

            rejected_source_fingerprint = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_source_fingerprint.returncode, 1)
            rejected_source_fingerprint_payload = json.loads(rejected_source_fingerprint.stdout)
            self.assertFalse(rejected_source_fingerprint_payload["valid"])
            self.assertIn(
                "invalid or missing pine_source_lock_fingerprint",
                "\n".join(rejected_source_fingerprint_payload["errors"]),
            )
            index_payload = json.loads(generated.stdout)
            index_payload["pine_source_locks"][0]["sha256"] = "1" * 64
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

            rejected_source_lock = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_source_lock.returncode, 1)
            rejected_source_lock_payload = json.loads(rejected_source_lock.stdout)
            self.assertFalse(rejected_source_lock_payload["valid"])
            self.assertIn(
                "pine_source_lock_fingerprint mismatch",
                "\n".join(rejected_source_lock_payload["errors"]),
            )
            index_payload = json.loads(generated.stdout)
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
            index_payload["exports"].append(dict(index_payload["exports"][0]))
            index_payload["summary"] = {"missing": 2, "present": 0, "total": 2}
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

            rejected_duplicate_export = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_duplicate_export.returncode, 1)
            rejected_duplicate_export_payload = json.loads(rejected_duplicate_export.stdout)
            self.assertFalse(rejected_duplicate_export_payload["valid"])
            duplicate_export_errors = "\n".join(rejected_duplicate_export_payload["errors"])
            self.assertIn("duplicate export row: missing_case", duplicate_export_errors)
            self.assertIn("duplicate export target_csv: missing.csv", duplicate_export_errors)
            self.assertIn("duplicate helper snippet path", duplicate_export_errors)
            index_payload = json.loads(generated.stdout)
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
            index_payload["summary"]["missing"] = 99
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

            rejected_summary = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_summary.returncode, 1)
            rejected_summary_payload = json.loads(rejected_summary.stdout)
            self.assertFalse(rejected_summary_payload["valid"])
            self.assertIn("summary missing mismatch", "\n".join(rejected_summary_payload["errors"]))
            index_payload["summary"]["missing"] = 1
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
            acceptance_path = output_dir / "acceptance_manifest.csv"
            acceptance_text = acceptance_path.read_text()
            with acceptance_path.open(newline="") as handle:
                duplicate_acceptance_rows = list(csv.DictReader(handle))
            duplicate_acceptance_rows.append(
                {
                    **duplicate_acceptance_rows[0],
                    "name": "duplicate_acceptance_case",
                }
            )
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(duplicate_acceptance_rows[0]))
                writer.writeheader()
                writer.writerows(duplicate_acceptance_rows)

            rejected_duplicate_acceptance = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_duplicate_acceptance.returncode, 1)
            rejected_duplicate_acceptance_payload = json.loads(rejected_duplicate_acceptance.stdout)
            self.assertFalse(rejected_duplicate_acceptance_payload["valid"])
            duplicate_acceptance_errors = "\n".join(rejected_duplicate_acceptance_payload["errors"])
            self.assertIn("duplicate acceptance target_csv: missing.csv", duplicate_acceptance_errors)
            self.assertIn("duplicate acceptance helper_snippet: missing_case.pine", duplicate_acceptance_errors)
            acceptance_path.write_text(acceptance_text)
            with acceptance_path.open(newline="") as handle:
                header_tamper_rows = list(csv.DictReader(handle))
            header_tamper_fieldnames = [field for field in list(header_tamper_rows[0]) if field != "proves"]
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header_tamper_fieldnames)
                writer.writeheader()
                for row in header_tamper_rows:
                    writer.writerow({field: row[field] for field in header_tamper_fieldnames})

            rejected_acceptance_header = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_acceptance_header.returncode, 1)
            rejected_acceptance_header_payload = json.loads(rejected_acceptance_header.stdout)
            self.assertFalse(rejected_acceptance_header_payload["valid"])
            self.assertIn(
                "acceptance manifest header mismatch",
                "\n".join(rejected_acceptance_header_payload["errors"]),
            )
            acceptance_path.write_text(acceptance_text)
            index_payload = json.loads(generated.stdout)
            index_payload["exports"][0]["settings_fingerprint"] = None
            index_payload["exports"][0]["settings_fingerprint_column"] = ""
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
            with acceptance_path.open(newline="") as handle:
                tampered_fingerprint_rows = list(csv.DictReader(handle))
            tampered_fingerprint_rows[0]["settings_fingerprint"] = "None"
            tampered_fingerprint_rows[0]["settings_fingerprint_column"] = ""
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(tampered_fingerprint_rows[0]))
                writer.writeheader()
                writer.writerows(tampered_fingerprint_rows)

            rejected_fingerprint_metadata = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_fingerprint_metadata.returncode, 1)
            rejected_fingerprint_metadata_payload = json.loads(rejected_fingerprint_metadata.stdout)
            self.assertFalse(rejected_fingerprint_metadata_payload["valid"])
            fingerprint_metadata_errors = "\n".join(rejected_fingerprint_metadata_payload["errors"])
            self.assertIn("invalid or missing settings_fingerprint", fingerprint_metadata_errors)
            self.assertIn("invalid or missing settings_fingerprint_column", fingerprint_metadata_errors)
            index_path.write_text(generated.stdout)
            acceptance_path.write_text(acceptance_text)
            index_payload = json.loads(generated.stdout)
            index_payload["exports"][0]["settings"] = {}
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
            with acceptance_path.open(newline="") as handle:
                tampered_settings_rows = list(csv.DictReader(handle))
            tampered_settings_rows[0]["settings_json"] = "{}"
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(tampered_settings_rows[0]))
                writer.writeheader()
                writer.writerows(tampered_settings_rows)

            rejected_settings_metadata = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_settings_metadata.returncode, 1)
            rejected_settings_metadata_payload = json.loads(rejected_settings_metadata.stdout)
            self.assertFalse(rejected_settings_metadata_payload["valid"])
            self.assertIn(
                "settings_fingerprint does not match settings",
                "\n".join(rejected_settings_metadata_payload["errors"]),
            )
            index_path.write_text(generated.stdout)
            acceptance_path.write_text(acceptance_text)
            index_payload = json.loads(generated.stdout)
            index_payload["exports"][0]["cli_flags"] = []
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")
            with acceptance_path.open(newline="") as handle:
                tampered_cli_rows = list(csv.DictReader(handle))
            tampered_cli_rows[0]["cli_flags"] = ""
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(tampered_cli_rows[0]))
                writer.writeheader()
                writer.writerows(tampered_cli_rows)

            rejected_cli_metadata = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_cli_metadata.returncode, 1)
            rejected_cli_metadata_payload = json.loads(rejected_cli_metadata.stdout)
            self.assertFalse(rejected_cli_metadata_payload["valid"])
            self.assertIn(
                "cli_flags do not match settings",
                "\n".join(rejected_cli_metadata_payload["errors"]),
            )
            index_path.write_text(generated.stdout)
            acceptance_path.write_text(acceptance_text)
            index_payload = json.loads(generated.stdout)
            index_payload["exports"][0]["tradingview_settings"] = []
            index_payload["exports"][0]["minimum_export_columns"] = []
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

            rejected_derived_settings_metadata = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_derived_settings_metadata.returncode, 1)
            rejected_derived_settings_metadata_payload = json.loads(rejected_derived_settings_metadata.stdout)
            self.assertFalse(rejected_derived_settings_metadata_payload["valid"])
            derived_settings_errors = "\n".join(rejected_derived_settings_metadata_payload["errors"])
            self.assertIn("tradingview_settings do not match settings", derived_settings_errors)
            self.assertIn("minimum_export_columns do not match settings", derived_settings_errors)
            index_path.write_text(generated.stdout)
            acceptance_path.write_text(acceptance_text)
            index_payload = json.loads(generated.stdout)
            index_payload["exports"][0]["pine_export_series"] = []
            index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n")

            rejected_pine_series_metadata = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_pine_series_metadata.returncode, 1)
            rejected_pine_series_metadata_payload = json.loads(rejected_pine_series_metadata.stdout)
            self.assertFalse(rejected_pine_series_metadata_payload["valid"])
            self.assertIn(
                "pine_export_series do not match settings",
                "\n".join(rejected_pine_series_metadata_payload["errors"]),
            )
            index_path.write_text(generated.stdout)
            acceptance_path.write_text(acceptance_text)
            with acceptance_path.open(newline="") as handle:
                tampered_rows = list(csv.DictReader(handle))
            tampered_rows[0]["required_full_export_columns"] = tampered_rows[0][
                "required_full_export_columns"
            ].replace("Settings Fingerprint", "Wrong Fingerprint")
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(tampered_rows[0]))
                writer.writeheader()
                writer.writerows(tampered_rows)

            rejected_columns = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_columns.returncode, 1)
            rejected_columns_payload = json.loads(rejected_columns.stdout)
            self.assertFalse(rejected_columns_payload["valid"])
            self.assertIn(
                "acceptance required_full_export_columns mismatch",
                "\n".join(rejected_columns_payload["errors"]),
            )
            acceptance_path.write_text(acceptance_text)
            with acceptance_path.open(newline="") as handle:
                tampered_metadata_rows = list(csv.DictReader(handle))
            tampered_metadata_rows[0]["settings_json"] = "{}"
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(tampered_metadata_rows[0]))
                writer.writeheader()
                writer.writerows(tampered_metadata_rows)

            rejected_metadata = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_metadata.returncode, 1)
            rejected_metadata_payload = json.loads(rejected_metadata.stdout)
            self.assertFalse(rejected_metadata_payload["valid"])
            self.assertIn(
                "acceptance settings_json mismatch",
                "\n".join(rejected_metadata_payload["errors"]),
            )
            acceptance_path.write_text(acceptance_text)
            index_path.write_text(generated.stdout)
            consistent_index_payload = json.loads(generated.stdout)
            helper_path = Path(consistent_index_payload["exports"][0]["helper_snippet"])
            helper_path.write_text(helper_path.read_text().replace("StopBuy", "StopBuyEdited", 1))
            edited_helper_sha = file_sha256(helper_path)
            consistent_index_payload["exports"][0]["helper_snippet_sha256"] = edited_helper_sha
            index_path.write_text(json.dumps(consistent_index_payload, indent=2, sort_keys=True) + "\n")
            with acceptance_path.open(newline="") as handle:
                helper_metadata_rows = list(csv.DictReader(handle))
            helper_metadata_rows[0]["helper_snippet_sha256"] = edited_helper_sha
            with acceptance_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(helper_metadata_rows[0]))
                writer.writeheader()
                writer.writerows(helper_metadata_rows)

            rejected_helper_content = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_helper_content.returncode, 1)
            rejected_helper_content_payload = json.loads(rejected_helper_content.stdout)
            self.assertFalse(rejected_helper_content_payload["valid"])
            self.assertIn(
                "helper snippet content does not match export metadata",
                "\n".join(rejected_helper_content_payload["errors"]),
            )
            index_path.write_text(generated.stdout)
            acceptance_path.write_text(acceptance_text)
            readme_path = output_dir / "README.md"
            readme_text = readme_path.read_text()
            readme_path.write_text(readme_text.replace("Export chart data", "Export edited chart data", 1))

            rejected_readme = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(rejected_readme.returncode, 1)
            rejected_readme_payload = json.loads(rejected_readme.stdout)
            self.assertFalse(rejected_readme_payload["valid"])
            self.assertIn(
                "export pack README content does not match export metadata",
                "\n".join(rejected_readme_payload["errors"]),
            )
            readme_path.write_text(readme_text)
            helper_path = Path(json.loads(generated.stdout)["exports"][0]["helper_snippet"])
            helper_path.write_text(helper_path.read_text() + "// tampered\n")

            rejected = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-export-pack",
                    str(output_dir),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(rejected.returncode, 1)
        rejection_payload = json.loads(rejected.stdout)
        self.assertFalse(rejection_payload["valid"])
        self.assertIn("helper snippet sha256 mismatch", "\n".join(rejection_payload["errors"]))

    def test_export_pack_rejects_stale_helper_snippets_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["stale export helper protection"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )
            output_dir = Path(tmp) / "pack"
            output_dir.mkdir()
            stale_path = output_dir / "old_case.pine"
            stale_path.write_text(EXPORT_PACK_HEADER + "\n// old helper\n")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("stale helper snippets", completed.stdout)
            self.assertIn(str(stale_path), completed.stdout)
            self.assertTrue(stale_path.exists())

    def test_export_pack_rejects_invalid_pine_source_locks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "sha256": "0" * 64,
                                "allow_debug_markers": False,
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "missing.csv",
                                "proves": ["source lock validation"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(Path(tmp) / "pack"),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("export pack requires valid manifest-pinned Pine sources", completed.stdout)
        self.assertIn("lcv6: missing PineScript/Indicators/lcv6.pine", completed.stdout)

    def test_export_pack_clean_stale_removes_only_generated_snippets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            (fixture_dir / "present.csv").write_text("time,open,high,low,close\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "present.csv",
                                "proves": ["stale export helper cleanup"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )
            output_dir = Path(tmp) / "pack"
            output_dir.mkdir()
            stale_path = output_dir / "old_case.pine"
            stale_path.write_text(EXPORT_PACK_HEADER + "\n// old helper\n")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                    "--clean-stale",
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertFalse(stale_path.exists())
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["summary"], {"missing": 1, "present": 0, "total": 1})
            self.assertTrue((output_dir / "missing_case.pine").exists())

            manual_path = output_dir / "manual.pine"
            manual_path.write_text("// manually maintained file\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_dir),
                    "--case",
                    "missing_case",
                    "--clean-stale",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("refusing to remove non export-pack Pine file", completed.stdout)
            self.assertTrue(manual_path.exists())

    def test_pine_export_helper_emits_reproducible_plot_snippet(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "lorentzian_classification",
                "pine-export-helper",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('plot(featureSeries.f1, "F1_RSI", display=display.data_window)', completed.stdout)
        self.assertIn('plot(prediction, "Prediction", display=display.data_window)', completed.stdout)
        self.assertIn('plot(startLongTrade ? 1 : na, "Buy", display=display.data_window)', completed.stdout)
        self.assertIn('plot(endShortTrade and settings.showExits ? 1 : na, "StopSell"', completed.stdout)
        self.assertNotIn("Kernel Plot Color", completed.stdout)

    def test_pine_export_helper_json_reports_full_export_modes(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "lorentzian_classification",
                "pine-export-helper",
                "--full",
                "--json",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["full"])
        self.assertEqual(payload["summary"]["total"], len(RESULT_FIELDNAMES))
        self.assertEqual(payload["summary"]["encoded_helper_required"], 0)
        self.assertGreater(payload["summary"]["encoded_plot_data_window"], 0)
        self.assertEqual(
            set(full_numeric_export_columns_for_settings(Settings())),
            set(pine_export_columns_for_settings(Settings(), include_full=True)),
        )
        self.assertEqual(len(full_numeric_export_columns_for_settings(Settings())), len(RESULT_FIELDNAMES))
        self.assertEqual(
            {row["column"] for row in payload["rows"]},
            set(RESULT_FIELDNAMES),
        )
        self.assertTrue(
            all(
                line.startswith("plot(") or line.startswith("//")
                for line in payload["plot_lines"]
            ),
            payload["plot_lines"],
        )
        self.assertTrue(
            all("Source expression" not in line for line in payload["plot_lines"]),
            payload["plot_lines"],
        )
        rows = {row["column"]: row for row in payload["rows"]}
        self.assertEqual(rows["Kernel Plot Color"]["export_mode"], "encoded_plot_data_window")
        self.assertEqual(rows["Trade Stats Visible"]["pine_expression"], "showTradeStats ? 1 : 0")
        self.assertEqual(rows["Table WL Ratio"]["pine_expression"], "totalWins / totalLosses")
        self.assertIn(
            'plot(backTestStream, "Backtest Stream", display=display.data_window)',
            payload["plot_lines"],
        )
        self.assertIn(
            'plot(showTradeStats ? 1 : 0, "Trade Stats Visible", display=display.data_window)',
            payload["plot_lines"],
        )
        self.assertTrue(
            any(line.startswith('plot(showKernelEstimate ?') for line in payload["plot_lines"]),
            payload["plot_lines"],
        )

    def test_pine_export_helper_uses_manifest_case_settings(self) -> None:
        manifest_path = ROOT / "tests" / "parity" / "fixtures_manifest.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "lorentzian_classification",
                "pine-export-helper",
                "--manifest",
                str(manifest_path),
                "--manifest-case",
                "coinbase_daily_custom_features_count3",
                "--full",
                "--json",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["manifest_case"], "coinbase_daily_custom_features_count3")
        self.assertEqual(payload["settings"]["feature_count"], 3)
        self.assertIn("F1_CCI", payload["minimum_export_columns"])
        self.assertIn("F2_RSI", payload["minimum_export_columns"])
        self.assertIn("F3_WT", payload["minimum_export_columns"])
        manifest_settings = settings_from_mapping(payload["settings"])
        self.assertEqual(payload["settings_fingerprint"], settings_fingerprint(manifest_settings))
        self.assertEqual(payload["settings_fingerprint_column"], SETTINGS_FINGERPRINT_COLUMN)
        self.assertIn(SETTINGS_FINGERPRINT_COLUMN, payload["full_instrumented_export_columns"])
        self.assertIn('plot(featureSeries.f1, "F1_CCI", display=display.data_window)', payload["plot_lines"])
        self.assertIn('plot(featureSeries.f2, "F2_RSI", display=display.data_window)', payload["plot_lines"])
        self.assertIn('plot(featureSeries.f3, "F3_WT", display=display.data_window)', payload["plot_lines"])
        self.assertIn(
            f'plot({payload["settings_fingerprint"]}, "Settings Fingerprint", display=display.data_window)',
            payload["plot_lines"],
        )

    def test_manifest_rejects_duplicate_required_case_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {"name": "duplicate", "filename": "a.csv", "settings": {}},
                            {"name": "duplicate", "filename": "b.csv", "settings": {}},
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("duplicate name values: duplicate", completed.stdout)

    def test_manifest_rejects_duplicate_external_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "external_sources": [
                            {"name": "main", "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src"},
                            {"name": "copy", "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src"},
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("external_sources: duplicate path values", completed.stdout)

    def test_manifest_rejects_case_insensitive_fixture_filename_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {"name": "upper", "filename": "Evidence.csv", "settings": {}},
                            {"name": "lower", "filename": "evidence.csv", "settings": {}},
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "fixtures: case-insensitive duplicate filename values: Evidence.csv, evidence.csv",
            completed.stdout,
        )
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_manifest_rejects_case_insensitive_source_path_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "pine_sources": [
                            {"name": "main", "path": "PineScript/Indicators/lcv6.pine"},
                            {"name": "copy", "path": "pinescript/indicators/lcv6.pine"},
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "pine_sources: case-insensitive duplicate path values: "
            "PineScript/Indicators/lcv6.pine, pinescript/indicators/lcv6.pine",
            completed.stdout,
        )
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_manifest_rejects_fixture_filename_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "escape",
                                "filename": "../outside.csv",
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid parity manifest", completed.stdout)
        self.assertIn("fixture 1: filename must not contain '..'", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_manifest_rejects_fixture_filename_windows_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "escape",
                                "filename": "C:\\Users\\export.csv",
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid parity manifest", completed.stdout)
        self.assertIn("fixture 1: filename must be a relative path", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_manifest_rejects_source_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "pine_sources": [
                            {
                                "name": "escape",
                                "path": "../outside.pine",
                                "role": "escaped source",
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid parity manifest", completed.stdout)
        self.assertIn("pine_sources 1: path must not contain '..'", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_manifest_rejects_source_windows_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "external_sources": [
                            {
                                "name": "escape",
                                "path": "D:\\external_runtime 5\\external\\indicator.external_src",
                                "role": "escaped source",
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("invalid parity manifest", completed.stdout)
        self.assertIn("external_sources 1: path must be a relative path", completed.stdout)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_manifest_rejects_tracked_and_required_filename_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [{"name": "tracked", "filename": "shared.csv", "settings": {}}],
                        "required_uncovered_fixture_cases": [
                            {"name": "required", "filename": "shared.csv", "settings": {}}
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-checklist",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("fixtures and required_uncovered_fixture_cases share filenames: shared.csv", completed.stdout)

    def test_manifest_rejects_ambiguous_required_case_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {"name": "case_a", "filename": "case_b", "settings": {}},
                            {"name": "case_b", "filename": "case_b.csv", "settings": {}},
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "pine-export-helper",
                    "--manifest",
                    str(manifest_path),
                    "--manifest-case",
                    "case_b",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("ambiguous name/filename selectors: case_b", completed.stdout)

    def test_manifest_rejects_label_only_required_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": ["planned dynamic exit export"],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-checklist",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("required_uncovered_fixture_cases 1: expected object", completed.stdout)

    def test_manifest_rejects_required_cases_without_proof_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "planned",
                                "filename": "planned.csv",
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-checklist",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("planned (planned.csv): proves must be a non-empty list", completed.stdout)

    def test_manifest_rejects_required_cases_without_smoke_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "planned",
                                "filename": "planned.csv",
                                "proves": ["planned export parity"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-checklist",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "planned (planned.csv): python_smoke_fixture must be a non-empty string",
            completed.stdout,
        )

    def test_manifest_rejects_required_case_smoke_fixture_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp) / "fixtures"
            fixture_dir.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "planned",
                                "filename": "planned.csv",
                                "python_smoke_fixture": "../known.csv",
                                "proves": ["planned export parity"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-checklist",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "planned (planned.csv): python_smoke_fixture must not contain '..'",
            completed.stdout,
        )
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_manifest_uncovered_cases_are_actionable_fixture_specs(self) -> None:
        manifest_path = ROOT / "tests" / "parity" / "fixtures_manifest.json"
        payload = json.loads(manifest_path.read_text())
        cases = payload["required_uncovered_fixture_cases"]
        self.assertTrue(cases)
        for case in cases:
            self.assertIsInstance(case["name"], str)
            self.assertTrue(case["name"])
            self.assertIsInstance(case["filename"], str)
            self.assertTrue(case["filename"].endswith(".csv"))
            self.assertIsInstance(case["python_smoke_fixture"], str)
            self.assertTrue(case["python_smoke_fixture"].endswith(".csv"))
            self.assertIsInstance(case["proves"], list)
            self.assertTrue(case["proves"])
            self.assertIsInstance(case["settings"], dict)
            # The same settings parser used by validate-fixtures must accept
            # every planned export case before the Pine CSV exists.
            self.assertIsInstance(settings_from_mapping(case["settings"]), Settings)

    def test_required_uncovered_settings_execute_against_representative_fixtures(self) -> None:
        manifest_path = ROOT / "tests" / "parity" / "fixtures_manifest.json"
        payload = load_manifest_payload(manifest_path)
        smoke_records = required_settings_smoke_records(payload, [self.fixture_dir, self.fixture_dir.parent])
        self.assertTrue(smoke_records)
        missing = [record for record in smoke_records if not record["present"]]
        if missing:
            raise unittest.SkipTest(f"Python smoke fixture not found: {missing[0]['python_smoke_fixture']}")
        failing = [record for record in smoke_records if not record["passed"]]
        self.assertFalse(failing, failing[:3])
        for record in smoke_records:
            self.assertGreater(record["rows"], 0)
            self.assertEqual(record["rows"], record["python_rows"])
            self.assertEqual(record["invariant_failures"], [])
            output_summary = record["output_summary"]
            self.assertIsInstance(output_summary, dict)
            self.assertTrue(output_summary["row_count_matches"])
            self.assertEqual(output_summary["bar_alignment_mismatches"], 0)
            self.assertEqual(output_summary["invalid_prediction_count"], 0)
            self.assertEqual(output_summary["invalid_direction_count"], 0)
            self.assertEqual(output_summary["invalid_backtest_stream_count"], 0)
            self.assertEqual(output_summary["trade_stats_visible_mismatches"], 0)
            self.assertGreater(output_summary["finite_kernel_count"], 0)
            for count in output_summary["active_feature_counts"].values():
                self.assertGreater(count, 0)

    def test_required_uncovered_settings_are_behaviorally_wired(self) -> None:
        manifest_path = ROOT / "tests" / "parity" / "fixtures_manifest.json"
        payload = load_manifest_payload(manifest_path)
        required_diff_fields = {
            "coinbase_daily_show_exits": {"stop_buy_count", "stop_sell_count"},
            "coinbase_daily_dynamic_exits": {"exit_buy_count", "exit_sell_count", "total_trades"},
            "oanda_daily_adx_filter": {"direction_sum", "buy_count", "sell_count"},
            "oanda_daily_ema_filter": {"buy_count", "sell_count"},
            "oanda_daily_sma_filter": {"buy_count", "sell_count"},
            "oanda_daily_kernel_smoothing": {"buy_count", "sell_count"},
            "coinbase_daily_hlc3_source": {"prediction_sum", "last_kernel"},
            "coinbase_daily_custom_features_count3": {"prediction_sum", "last_f1", "last_f2", "last_f3"},
            "oanda_daily_alternate_kernel_params": {"last_kernel", "sell_count"},
            "coinbase_daily_worst_case": {"total_wins", "total_losses", "win_loss_ratio"},
            "coinbase_daily_hide_trade_stats": {"trade_stats_visible"},
        }
        cases = payload["required_uncovered_fixture_cases"]
        self.assertEqual({case["name"] for case in cases}, set(required_diff_fields))

        for case in cases:
            fixture_path = self.fixture_dir / case["python_smoke_fixture"]
            if not fixture_path.exists():
                raise unittest.SkipTest(f"Python smoke fixture not found: {fixture_path}")
            tv_rows, price_scale = read_tradingview_csv(fixture_path)
            settings = settings_from_mapping(case["settings"])
            baseline = Settings(include_full_history=settings.include_full_history)

            baseline_signature = output_signature(calculate(tv_rows, settings=baseline, price_scale=price_scale))
            case_signature = output_signature(calculate(tv_rows, settings=settings, price_scale=price_scale))
            changed_fields = {
                key for key, baseline_value in baseline_signature.items() if baseline_value != case_signature[key]
            }

            missing = required_diff_fields[case["name"]] - changed_fields
            self.assertFalse(missing, f"{case['name']} did not change expected fields: {sorted(missing)}")

    def test_manifest_pine_sources_are_actionable_and_non_debug_canonical(self) -> None:
        manifest_path = ROOT / "tests" / "parity" / "fixtures_manifest.json"
        payload = load_manifest_payload(manifest_path)
        specs = load_pine_source_specs(payload)
        self.assertTrue(specs)
        canonical = [spec for spec in specs if spec.name == "lcv6"]
        self.assertEqual(len(canonical), 1)
        canonical_path = DEFAULT_FIXTURE_DIR.parent / canonical[0].path
        if not canonical_path.exists():
            raise unittest.SkipTest(f"Pine source not found: {canonical_path}")
        self.assertFalse(has_debug_markers(canonical_path))
        records = pine_source_records([DEFAULT_FIXTURE_DIR, DEFAULT_FIXTURE_DIR.parent], specs)
        self.assertTrue(records)
        for record in records:
            self.assertTrue(record["valid"], record)
            self.assertTrue(record["present"], record)
            self.assertFalse(record["debug_markers"], record)
            self.assertTrue(
                record["sha256_matches"] is True or record["code_sha256_matches"] is True,
                record,
            )

    def test_manifest_source_code_hash_allows_comment_only_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            pine_dir = root / "PineScript" / "Indicators"
            pine_dir.mkdir(parents=True)
            source = pine_dir / "lcv6.pine"
            source.write_text("// changed comment\nindicator(\"fixture\")\n\nplot(close)\n")
            payload = {
                "pine_sources": [
                    {
                        "name": "lcv6",
                        "path": "PineScript/Indicators/lcv6.pine",
                        "role": "canonical",
                        "sha256": "0" * 64,
                        "code_sha256": source_code_sha256(source),
                        "allow_debug_markers": False,
                    }
                ]
            }

            records, errors = validate_source_records([root], load_pine_source_specs(payload))

        self.assertFalse(records[0]["sha256_matches"])
        self.assertTrue(records[0]["code_sha256_matches"])
        self.assertTrue(records[0]["valid"], records[0])
        self.assertEqual(errors, [])

    def test_manifest_source_debug_policy_catches_external_logging_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            source = root / "Indicators" / "LorentzianClassification" / "LorentzianClassification.external_src"
            source.parent.mkdir(parents=True)
            source.write_text(
                "\n".join(
                    [
                        "void OnStart()",
                        "{",
                        '   PrintFormat("prediction=%d", 1);',
                        '   Comment("parity check running");',
                        '   Alert("parity check complete");',
                        "}",
                    ]
                )
            )
            payload = {
                "external_sources": [
                    {
                        "name": "LorentzianClassification",
                        "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                        "role": "canonical",
                        "sha256": file_sha256(source),
                        "allow_debug_markers": False,
                    }
                ]
            }

            records, errors = validate_source_records([root], load_external_source_specs(payload))

        self.assertTrue(records[0]["sha256_matches"], records[0])
        self.assertTrue(records[0]["debug_markers"], records[0])
        self.assertFalse(records[0]["valid"], records[0])
        self.assertEqual(errors, ["LorentzianClassification: debug markers are not allowed"])

    def test_manifest_source_directories_are_missing_source_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            root.mkdir()
            pine_source_path = root / "PineScript" / "Indicators" / "lcv6.pine"
            pine_source_path.mkdir(parents=True)
            external_source_path = root / "Indicators" / "LorentzianClassification" / "LorentzianClassification.external_src"
            external_source_path.mkdir(parents=True)
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "canonical Pine source",
                                "allow_debug_markers": False,
                            }
                        ],
                        "external_sources": [
                            {
                                "name": "LorentzianClassification",
                                "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                                "role": "canonical external source",
                                "allow_debug_markers": False,
                            }
                        ],
                        "fixtures": [],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_case",
                                "filename": "missing.csv",
                                "python_smoke_fixture": "missing.csv",
                                "proves": ["source directory validation"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )
            payload = load_manifest_payload(manifest_path)
            pine_records = pine_source_records([root], load_pine_source_specs(payload))
            external_records = source_records([root], load_external_source_specs(payload))

            audit = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "audit-fixtures",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            export_pack = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "export-pack",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(Path(tmp) / "pack"),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertFalse(pine_records[0]["present"])
        self.assertFalse(pine_records[0]["valid"])
        self.assertIsNone(pine_records[0]["resolved_path"])
        self.assertFalse(external_records[0]["present"])
        self.assertFalse(external_records[0]["valid"])
        self.assertIsNone(external_records[0]["resolved_path"])
        self.assertEqual(audit.returncode, 0, audit.stdout + audit.stderr)
        self.assertNotIn("Traceback", audit.stdout + audit.stderr)
        audit_payload = json.loads(audit.stdout)
        self.assertFalse(audit_payload["manifest_pine_sources"][0]["present"])
        self.assertFalse(audit_payload["manifest_external_sources"][0]["present"])
        self.assertEqual(export_pack.returncode, 1)
        self.assertIn("export pack requires valid manifest-pinned Pine sources", export_pack.stdout)
        self.assertIn("lcv6: missing PineScript/Indicators/lcv6.pine", export_pack.stdout)
        self.assertNotIn("Traceback", export_pack.stdout + export_pack.stderr)

    def test_repo_pinescript_reference_matches_manifest_pins(self) -> None:
        manifest_path = ROOT / "tests" / "parity" / "fixtures_manifest.json"
        payload = load_manifest_payload(manifest_path)
        expected_sha = {spec.name: spec.sha256 for spec in load_pine_source_specs(payload)}
        expected_code_sha = {spec.name: spec.code_sha256 for spec in load_pine_source_specs(payload)}
        repo_sources = {
            "lcv6": REPO_PINE_LCV6,
            "MLExtensions": REPO_PINE_ML_EXTENSIONS,
            "KernelFunctions": REPO_PINE_KERNEL_FUNCTIONS,
        }
        for name, path in repo_sources.items():
            with self.subTest(name=name):
                self.assertTrue(path.exists(), f"repo Pine source missing: {path}")
                source = path.read_text()
                self.assertNotIn("Placeholder", source)
                self.assertFalse(has_debug_markers(path))
                if file_sha256(path) != expected_sha[name]:
                    self.assertEqual(source_code_sha256(path), expected_code_sha[name])
                else:
                    self.assertEqual(file_sha256(path), expected_sha[name])

    def test_repo_status_docs_reflect_python_and_pine_parity_state(self) -> None:
        readme = (ROOT / "README.md").read_text()
        validation = (ROOT / "docs" / "validation.md").read_text()
        tests_readme = (ROOT / "tests" / "README.md").read_text()
        coverage = (ROOT / "tests" / "parity" / "python_port_coverage.md").read_text()

        self.assertIn("ports/pinescript/", readme)
        self.assertIn("SHA-pinned", readme)
        self.assertIn("Python CLI and library port", readme)
        self.assertIn("strict non-default coverage is still expanding", readme)
        self.assertNotIn("Placeholder", readme)
        self.assertNotIn("Pine Script", readme)
        self.assertIn("Python parity artifacts have been added", validation)
        self.assertNotIn("No validation artifacts have been imported", validation)
        self.assertIn("tests/parity/", tests_readme)
        self.assertIn("Settings Fingerprint", coverage)
        self.assertIn("deterministic numeric helper codes", coverage)
        self.assertNotIn("not mandatory plain CSV columns", coverage)
        self.assertNotIn("helper-only comments", coverage)

    def test_repo_baselines_manifest_registers_only_standard_python_fixtures(self) -> None:
        specs = load_parity_manifest(BASELINES_MANIFEST)
        filenames = {spec.filename for spec in specs}
        self.assertEqual(
            filenames,
            {
                "pine_oanda_eurusd_1d_full_history.csv",
                "pine_tastyfx_eurusd_1d_full_history.csv",
                "pine_coinbase_btcusd_1d_limited_history.csv",
                "pine_btcusd_h1_trimmed_limited_history.csv",
            },
        )
        for spec in specs:
            with self.subTest(filename=spec.filename):
                fixture_path = BASELINES_DIR / spec.filename
                self.assertTrue(fixture_path.exists(), fixture_path)
                schema = pine_export_schema(fixture_path, spec.settings)
                self.assertTrue(schema["valid"], schema)
                self.assertEqual(spec.tolerance, 1e-6)

    def test_repo_manifest_defers_external_sources_until_added_later(self) -> None:
        manifest_path = ROOT / "tests" / "parity" / "fixtures_manifest.json"
        payload = load_manifest_payload(manifest_path)
        self.assertEqual(load_external_source_specs(payload), [])
        self.assertEqual(load_external_parity_report_specs(payload), [])

    def test_external_compiled_artifact_flags_stale_or_missing_ex5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            indicator = root / "Indicators" / "LorentzianClassification" / "LorentzianClassification.external_src"
            include = root / "Indicators" / "LorentzianClassification" / "Include" / "ANN.external_inc"
            indicator.parent.mkdir(parents=True)
            include.parent.mkdir(parents=True)
            indicator.write_text("// indicator\n")
            include.write_text("// include\n")
            records = [
                {
                    "name": "LorentzianClassification",
                    "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                    "present": True,
                    "resolved_path": str(indicator),
                },
                {
                    "name": "ANN",
                    "path": "Indicators/LorentzianClassification/Include/ANN.external_inc",
                    "present": True,
                    "resolved_path": str(include),
                },
            ]

            missing_records = external_compiled_artifact_records(records)
            self.assertEqual(len(missing_records), 1)
            self.assertFalse(missing_records[0]["passed"])
            self.assertFalse(missing_records[0]["present"])

            artifact = indicator.with_suffix(".compiled")
            artifact.write_text("compiled\n")
            os.utime(indicator, (2000, 2000))
            os.utime(include, (3000, 3000))
            os.utime(artifact, (2500, 2500))
            stale_records = external_compiled_artifact_records(records)
            self.assertFalse(stale_records[0]["passed"])
            self.assertIn("older than", stale_records[0]["detail"])

            os.utime(artifact, (4000, 4000))
            fresh_records = external_compiled_artifact_records(records)
            self.assertTrue(fresh_records[0]["passed"], fresh_records)
            self.assertEqual(fresh_records[0]["newest_source_path"], str(include))

    def test_external_parity_report_records_validate_matches_and_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "parity.csv"
            report.write_text(
                "\n".join(
                    [
                        "time,close,tv_f1,external_f1,f1_diff,tv_kernel,external_kernel,kernel_diff,"
                        "tv_prediction,external_prediction,pred_match,tv_direction,external_direction,dir_match,"
                        "tv_buy,external_buy,tv_sell,external_sell",
                        "1,1.0,0.5,0.5,0.0,1.0,1.0,0.0,3,3,1,1,1,1,,,,",
                        "2,1.1,0.6,0.6,0.0,1.1,1.1,0.0,2,2,1,1,1,1,1,1,,",
                    ]
                )
            )
            os.utime(report, (2000, 2000))
            payload = {
                "external_parity_reports": [
                    {
                        "name": "fixture_report",
                        "filename": "parity.csv",
                        "role": "unit test report",
                        "input_filename": "input.csv",
                        "include_full_history": False,
                        "required": True,
                    }
                ]
            }
            specs = load_external_parity_report_specs(payload)

            fresh = external_parity_report_records(
                [root],
                specs,
                [{"passed": True, "artifact_mtime": 1000.0}],
                1e-9,
            )[0]
            stale = external_parity_report_records(
                [root],
                specs,
                [{"passed": True, "artifact_mtime": 3000.0}],
                1e-9,
            )[0]

        self.assertTrue(fresh["passed"], fresh)
        self.assertEqual(fresh["rows"], 2)
        self.assertEqual(fresh["compared_prediction_rows"], 2)
        self.assertEqual(fresh["direction_mismatches"], 0)
        self.assertEqual(fresh["buy_mismatches"], 0)
        self.assertFalse(fresh["stale"])
        action = fresh["regeneration_action"]
        self.assertEqual(action["script_inputs"]["InpInputFile"], "input.csv")
        self.assertEqual(action["script_inputs"]["InpOutputFile"], "parity.csv")
        self.assertFalse(action["script_inputs"]["InpIncludeFullHist"])
        self.assertFalse(stale["passed"])
        self.assertTrue(stale["stale"])
        self.assertIn("older than the compiled indicator artifact", stale["error"])

    def test_external_parity_report_records_reject_signal_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "parity.csv"
            report.write_text(
                "\n".join(
                    [
                        "time,close,tv_f1,external_f1,f1_diff,tv_prediction,external_prediction,pred_match,"
                        "tv_direction,external_direction,dir_match,tv_buy,external_buy,tv_sell,external_sell",
                        "1,1.0,0.5,0.5,0.0,3,2,0,1,-1,0,1,,,"
                    ]
                )
            )
            payload = {
                "external_parity_reports": [
                    {
                        "name": "failing_report",
                        "filename": "parity.csv",
                        "role": "unit test report",
                        "input_filename": "input.csv",
                        "include_full_history": False,
                    }
                ]
            }
            record = external_parity_report_records(
                [root],
                load_external_parity_report_specs(payload),
                [],
                1e-9,
            )[0]

        self.assertFalse(record["passed"])
        self.assertEqual(record["prediction_mismatches"], 1)
        self.assertEqual(record["direction_mismatches"], 1)
        self.assertEqual(record["buy_mismatches"], 1)
        self.assertIn("prediction or direction mismatches", record["error"])

    def test_external_report_checklist_outputs_regeneration_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            files = root / "Files"
            indicator_dir = root / "Indicators" / "LorentzianClassification"
            scripts_dir = root / "Scripts"
            pine_dir = root / "PineScript" / "Indicators"
            files.mkdir(parents=True)
            indicator_dir.mkdir(parents=True)
            scripts_dir.mkdir(parents=True)
            pine_dir.mkdir(parents=True)
            pine_source = pine_dir / "lcv6.pine"
            pine_source.write_text("//@version=6\nindicator('lcv6')\n")
            known = files / "known.csv"
            known.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            report = files / "parity.csv"
            report.write_text(
                "\n".join(
                    [
                        "time,close,tv_f1,external_f1,f1_diff,tv_prediction,external_prediction,pred_match,"
                        "tv_direction,external_direction,dir_match,tv_buy,external_buy,tv_sell,external_sell",
                        "1,1.0,0.5,0.5,0.0,3,3,1,1,1,1,,,,",
                    ]
                )
            )
            indicator = indicator_dir / "LorentzianClassification.external_src"
            indicator.write_text("int begin = 0;\nbool filterAll = filtVol && filtRegime && filtAdx;\n")
            artifact = indicator.with_suffix(".compiled")
            artifact.write_text("compiled\n")
            script = scripts_dir / "LorentzianParityCheck.external_src"
            script.write_text("// parity script\n")
            script_artifact = script.with_suffix(".compiled")
            script_artifact.write_text("compiled script\n")
            os.utime(report, (2000, 2000))
            os.utime(indicator, (3000, 3000))
            os.utime(artifact, (4000, 4000))
            os.utime(script, (3500, 3500))
            os.utime(script_artifact, (4500, 4500))
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [{"name": "known", "filename": "known.csv", "settings": {}}],
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "unit test Pine source",
                            }
                        ],
                        "external_sources": [
                            {
                                "name": "LorentzianClassification",
                                "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                                "role": "canonical external",
                            }
                        ],
                        "external_parity_reports": [
                            {
                                "name": "fixture_report",
                                "filename": "parity.csv",
                                "role": "unit test report",
                                "input_filename": "input.csv",
                                "include_full_history": False,
                                "required": True,
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "external-report-checklist",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--case",
                    "input.csv",
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["summary"]["total"], 1)
        self.assertEqual(payload["summary"]["failed"], 1)
        self.assertEqual(payload["summary"]["stale"], 1)
        self.assertEqual(payload["summary"]["script_artifacts_failed"], 0)
        self.assertEqual(payload["external_parity_script_artifacts"][0]["script_path"], "Scripts/LorentzianParityCheck.external_src")
        self.assertTrue(payload["external_parity_script_artifacts"][0]["passed"])
        record = payload["reports"][0]
        self.assertEqual(record["name"], "fixture_report")
        self.assertTrue(record["stale"])
        action = record["regeneration_action"]
        self.assertEqual(action["script_inputs"]["InpInputFile"], "input.csv")
        self.assertEqual(action["script_inputs"]["InpOutputFile"], "parity.csv")
        self.assertFalse(action["script_inputs"]["InpIncludeFullHist"])

    def test_external_runner_pack_writes_terminal_startup_files_for_stale_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            files = root / "Files"
            indicator_dir = root / "Indicators" / "LorentzianClassification"
            scripts_dir = root / "Scripts"
            pine_dir = root / "PineScript" / "Indicators"
            files.mkdir(parents=True)
            indicator_dir.mkdir(parents=True)
            scripts_dir.mkdir(parents=True)
            pine_dir.mkdir(parents=True)
            pine_source = pine_dir / "lcv6.pine"
            pine_source.write_text("//@version=6\nindicator('lcv6')\n")
            known = files / "known.csv"
            known.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            report = files / "parity.csv"
            report.write_text(
                "\n".join(
                    [
                        "time,close,tv_f1,external_f1,f1_diff,tv_prediction,external_prediction,pred_match,"
                        "tv_direction,external_direction,dir_match,tv_buy,external_buy,tv_sell,external_sell",
                        "1,1.0,0.5,0.5,0.0,3,3,1,1,1,1,,,,",
                    ]
                )
            )
            indicator = indicator_dir / "LorentzianClassification.external_src"
            indicator.write_text("int begin = 0;\nbool filterAll = filtVol && filtRegime && filtAdx;\n")
            artifact = indicator.with_suffix(".compiled")
            artifact.write_text("compiled\n")
            script = scripts_dir / "LorentzianParityCheck.external_src"
            script.write_text("// parity script\n")
            script_artifact = script.with_suffix(".compiled")
            script_artifact.write_text("compiled script\n")
            os.utime(report, (2000, 2000))
            os.utime(indicator, (3000, 3000))
            os.utime(artifact, (4000, 4000))
            os.utime(script, (3500, 3500))
            os.utime(script_artifact, (4500, 4500))
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [{"name": "known", "filename": "known.csv", "settings": {}}],
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "unit test Pine source",
                            }
                        ],
                        "external_sources": [
                            {
                                "name": "LorentzianClassification",
                                "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                                "role": "canonical external",
                            }
                        ],
                        "external_parity_reports": [
                            {
                                "name": "fixture_report",
                                "filename": "parity.csv",
                                "role": "unit test report",
                                "input_filename": "input.csv",
                                "include_full_history": False,
                                "required": True,
                            },
                            {
                                "name": "other_report",
                                "filename": "other.csv",
                                "role": "unselected unit test report",
                                "input_filename": "other_input.csv",
                                "include_full_history": True,
                                "required": True,
                            },
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_export",
                                "filename": "missing_export.csv",
                                "python_smoke_fixture": "known.csv",
                                "proves": ["combined artifact preparation"],
                                "settings": {"show_exits": True},
                            }
                        ],
                    }
                )
            )
            output = Path(tmp) / "runner-pack"
            artifacts_output = Path(tmp) / "artifacts"
            staged_exports = Path(tmp) / "TradingView Downloads"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "external-runner-pack",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--case",
                    "fixture_report",
                    "--only-failing",
                    "--output",
                    str(output),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            prepare_completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "prepare-readiness-artifacts",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(artifacts_output),
                    "--pine-export-source-dir",
                    str(staged_exports),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            verify_readiness_completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-readiness-artifacts",
                    str(artifacts_output),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            readiness_manifest = json.loads((artifacts_output / "readiness_artifacts.json").read_text())
            readiness_readme = (artifacts_output / "README.md").read_text()
            (artifacts_output / "README.md").write_text("edited\n")
            corrupt_readiness_completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-readiness-artifacts",
                    str(artifacts_output),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            preset_path = output / "Presets" / "lorentzian_parity_fixture_report.set"
            startup_path = output / "Scripts" / "parity_debug" / "run_parity_fixture_report.ini"
            preset_text = preset_path.read_text()
            startup_text = startup_path.read_text()
            verify_completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-external-runner-pack",
                    str(output),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )
            preset_path.write_text(preset_text.replace("InpOutputFile=parity.csv", "InpOutputFile=wrong.csv"))
            corrupt_completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "verify-external-runner-pack",
                    str(output),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["record_count"], 1)
        record = payload["records"][0]
        self.assertEqual(record["name"], "fixture_report")
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["stale"])
        self.assertEqual(
            preset_text,
            "InpInputFile=input.csv\nInpOutputFile=parity.csv\nInpIncludeFullHist=false\n",
        )
        self.assertIn("Script=LorentzianParityCheck", startup_text)
        self.assertIn("ScriptParameters=lorentzian_parity_fixture_report.set", startup_text)
        self.assertEqual(
            record["terminal_command"],
            [
                "external_runtime.exe",
                "/portable",
                '/config:"external\\Scripts\\parity_debug\\run_parity_fixture_report.ini"',
            ],
        )
        self.assertEqual(verify_completed.returncode, 0, verify_completed.stdout + verify_completed.stderr)
        verify_payload = json.loads(verify_completed.stdout)
        self.assertTrue(verify_payload["valid"])
        self.assertEqual(verify_payload["record_count"], 1)
        self.assertEqual(prepare_completed.returncode, 0, prepare_completed.stdout + prepare_completed.stderr)
        prepare_payload = json.loads(prepare_completed.stdout)
        self.assertTrue(prepare_payload["valid"])
        self.assertTrue(prepare_payload["export_pack"]["verify"]["valid"])
        self.assertEqual(prepare_payload["export_pack"]["verify"]["export_count"], 1)
        self.assertTrue(prepare_payload["external_runner_pack"]["verify"]["valid"])
        self.assertEqual(prepare_payload["external_runner_pack"]["verify"]["record_count"], 2)
        self.assertEqual(prepare_payload["pine_export_source_dir"], str(staged_exports))
        self.assertEqual(readiness_manifest, prepare_payload)
        self.assertIn("Lorentzian Readiness Artifacts", readiness_readme)
        self.assertIn("verify-export-pack", readiness_readme)
        self.assertIn("verify-external-runner-pack", readiness_readme)
        self.assertIn("Import Downloaded Pine Exports", readiness_readme)
        self.assertIn(f"import-pine-exports {shlex.quote(str(staged_exports))}", readiness_readme)
        self.assertIn("readiness --fixture-dir", readiness_readme)
        self.assertEqual(verify_readiness_completed.returncode, 0, verify_readiness_completed.stdout + verify_readiness_completed.stderr)
        verify_readiness_payload = json.loads(verify_readiness_completed.stdout)
        self.assertTrue(verify_readiness_payload["valid"])
        self.assertEqual(verify_readiness_payload["export_pack"]["export_count"], 1)
        self.assertEqual(verify_readiness_payload["external_runner_pack"]["record_count"], 2)
        self.assertEqual(corrupt_readiness_completed.returncode, 1)
        corrupt_readiness_payload = json.loads(corrupt_readiness_completed.stdout)
        self.assertFalse(corrupt_readiness_payload["valid"])
        self.assertTrue(any("README" in error for error in corrupt_readiness_payload["errors"]))
        self.assertEqual(corrupt_completed.returncode, 1)
        corrupt_payload = json.loads(corrupt_completed.stdout)
        self.assertFalse(corrupt_payload["valid"])
        self.assertTrue(any("InpOutputFile" in error for error in corrupt_payload["errors"]))

    def test_external_parity_script_contract_tracks_report_regeneration_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            script_path = root / "Scripts" / "LorentzianParityCheck.external_src"
            script_path.parent.mkdir(parents=True)
            script_path.write_text(external_parity_script_contract_source())
            specs = [
                externalParityReportSpec(
                    name="fixture_report",
                    filename="parity.csv",
                    role="test report",
                    input_filename="input.csv",
                    include_full_history=False,
                )
            ]

            records = external_parity_script_contract_records([root], specs)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["passed"], records[0])
        self.assertEqual(records[0]["missing_inputs"], [])
        self.assertEqual(records[0]["missing_usage"], [])
        self.assertEqual(records[0]["missing_report_columns"], [])

    def test_external_parity_script_contract_flags_missing_regeneration_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            script_path = root / "Scripts" / "LorentzianParityCheck.external_src"
            script_path.parent.mkdir(parents=True)
            script_path.write_text(external_parity_script_contract_source(include_output_input=False))
            specs = [
                externalParityReportSpec(
                    name="fixture_report",
                    filename="parity.csv",
                    role="test report",
                    input_filename="input.csv",
                    include_full_history=False,
                )
            ]

            records = external_parity_script_contract_records([root], specs)

        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["passed"])
        self.assertEqual(records[0]["missing_inputs"], ["InpOutputFile"])
        self.assertTrue(any("InpOutputFile" in error for error in records[0]["errors"]))

    def test_external_indicator_input_contract_tracks_python_settings_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            indicator = Path(tmp) / "LorentzianClassification.external_src"
            indicator.write_text(external_indicator_input_contract_source())
            records = [
                {
                    "name": "LorentzianClassification",
                    "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                    "present": True,
                    "resolved_path": str(indicator),
                }
            ]

            contracts = external_indicator_input_contract_records(records)

        self.assertEqual(len(contracts), 1)
        self.assertTrue(contracts[0]["passed"], contracts[0])
        self.assertEqual(contracts[0]["missing"], [])
        self.assertEqual(contracts[0]["mismatches"], [])

    def test_external_indicator_input_contract_flags_pine_default_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            indicator = Path(tmp) / "LorentzianClassification.external_src"
            indicator.write_text(external_indicator_input_contract_source(use_atr_offset="false"))
            records = [
                {
                    "name": "LorentzianClassification",
                    "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                    "present": True,
                    "resolved_path": str(indicator),
                }
            ]

            contracts = external_indicator_input_contract_records(records)

        self.assertEqual(len(contracts), 1)
        self.assertFalse(contracts[0]["passed"])
        mismatch = contracts[0]["mismatches"][0]
        self.assertEqual(mismatch["setting"], "use_atr_offset")
        self.assertEqual(mismatch["input"], "InpUseAtrOffset")
        self.assertEqual(mismatch["expected"], "true")
        self.assertEqual(mismatch["actual"], "false")

    def test_external_indicator_input_contract_flags_missing_display_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            indicator = Path(tmp) / "LorentzianClassification.external_src"
            indicator.write_text(external_indicator_input_contract_source(include_confidence_gradient=False))
            records = [
                {
                    "name": "LorentzianClassification",
                    "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                    "present": True,
                    "resolved_path": str(indicator),
                }
            ]

            contracts = external_indicator_input_contract_records(records)

        self.assertEqual(len(contracts), 1)
        self.assertFalse(contracts[0]["passed"])
        self.assertIn("InpUseConfidenceGradient", {row["input"] for row in contracts[0]["missing"]})

    def test_external_source_sanity_flags_known_incremental_replay_traps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            indicator = Path(tmp) / "LorentzianClassification.external_src"
            indicator.write_text(
                "\n".join(
                    [
                        "int begin = (prev_calculated > 0) ? prev_calculated - 1 : 0;",
                        "if(prev_calculated == 0)",
                        "   InitANN(g_ann);",
                        "bool filterAll = filtVol && filtRegime && filtAdx;",
                    ]
                )
            )
            records = [
                {
                    "name": "LorentzianClassification",
                    "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                    "present": True,
                    "resolved_path": str(indicator),
                }
            ]

            sanity = external_source_sanity_records(records)

        failed = {(row["source"], row["check"]) for row in sanity if not row["passed"]}
        self.assertIn(("LorentzianClassification", "ann_replay_starts_from_bar_zero"), failed)
        self.assertIn(("LorentzianClassification", "ann_state_resets_each_calculation"), failed)
        self.assertNotIn(("LorentzianClassification", "signal_flip_filter_gate_matches_pine"), failed)

    def test_validate_fixtures_json_reports_pinned_pine_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            files = root / "Files"
            pine_dir = root / "PineScript" / "Indicators"
            files.mkdir(parents=True)
            pine_dir.mkdir(parents=True)
            fixture_path = files / "known.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,1.12,0,0,,,,",
                    ]
                )
            )
            (pine_dir / "lcv6.pine").write_text('indicator("drifted fixture")\n')
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "canonical",
                                "sha256": "0" * 64,
                                "allow_debug_markers": False,
                            }
                        ],
                        "fixtures": [
                            {
                                "name": "known",
                                "filename": "known.csv",
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["summary"]["pine_sources_invalid"], 1)
        self.assertFalse(payload["pine_sources"][0]["sha256_matches"])
        self.assertIn("sha256 mismatch", payload["source_errors"][0])

    def test_validate_fixtures_json_reports_pinned_external_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            files = root / "Files"
            indicator_dir = root / "Indicators" / "LorentzianClassification"
            files.mkdir(parents=True)
            indicator_dir.mkdir(parents=True)
            fixture_path = files / "known.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,1.12,0,0,,,,",
                    ]
                )
            )
            (indicator_dir / "LorentzianClassification.external_src").write_text("// drifted fixture\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "external_sources": [
                            {
                                "name": "LorentzianClassification",
                                "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                                "role": "canonical external",
                                "sha256": "0" * 64,
                                "allow_debug_markers": False,
                            }
                        ],
                        "fixtures": [
                            {
                                "name": "known",
                                "filename": "known.csv",
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["summary"]["external_sources_invalid"], 1)
        self.assertFalse(payload["external_sources"][0]["sha256_matches"])
        self.assertIn("sha256 mismatch", payload["source_errors"][0])

    def test_fixture_dir_accepts_external_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            files = root / "Files"
            files.mkdir(parents=True)
            self.assertEqual(fixture_dir_from_arg(str(root)), files)
            self.assertEqual(fixture_search_dirs_from_arg(str(root)), [files, root])

    def test_validate_fixtures_uses_fixture_dir_environment_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            files = root / "Files"
            files.mkdir(parents=True)
            fixture_path = files / "env_export.csv"
            fixture_path.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "env_export",
                                "filename": "env_export.csv",
                                "settings": {},
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(PYTHON_PORT),
                    FIXTURE_DIR_ENV_VAR: str(root),
                },
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn(f"  - {root / 'Files'}", completed.stdout)
        self.assertIn(f"  - {root}", completed.stdout)

    def test_validate_fixtures_searches_external_root_after_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            (root / "Files").mkdir(parents=True)
            root_fixture = root / "root_export.csv"
            root_fixture.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_RSI,F2_WT,F3_CCI,F4_ADX,F5_RSI9,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.10,1.20,1.00,1.15,,,,,,,0,0,,,,",
                    ]
                )
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "root_export",
                                "filename": "root_export.csv",
                                "settings": {},
                            }
                        ]
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn(f"  - {root / 'Files'}", completed.stdout)
        self.assertIn(f"  - {root}", completed.stdout)
        self.assertIn("PASS root_export.csv", completed.stdout)

    def test_audit_fixtures_inventories_workspace_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            files = root / "Files"
            pine_dir = root / "PineScript" / "Indicators"
            external_dir = root / "Indicators" / "LorentzianClassification"
            files.mkdir(parents=True)
            pine_dir.mkdir(parents=True)
            external_dir.mkdir(parents=True)
            (files / "known.csv").write_text("time,open,high,low,close\n")
            (root / "root_only.csv").write_text(
                "time,open,high,low,close,F1_RSI,Kernel Regression Estimate,Prediction,Direction,Buy,Sell\n"
            )
            (pine_dir / "lcv6.pine").write_text('indicator("fixture")\n')
            (external_dir / "LorentzianClassification.external_src").write_text("// indicator fixture\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "canonical",
                            }
                        ],
                        "external_sources": [
                            {
                                "name": "LorentzianClassification",
                                "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                                "role": "canonical external",
                            }
                        ],
                        "fixtures": [
                            {
                                "name": "known",
                                "filename": "known.csv",
                                "settings": {},
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_required",
                                "filename": "missing_required.csv",
                                "python_smoke_fixture": "known.csv",
                                "proves": ["audit required export inventory"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "audit-fixtures",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("PRESENT lcv6 (canonical) PineScript/Indicators/lcv6.pine", completed.stdout)
        self.assertIn(
            "PRESENT LorentzianClassification (canonical external) "
            "Indicators/LorentzianClassification/LorentzianClassification.external_src",
            completed.stdout,
        )
        self.assertIn("debug_markers=no", completed.stdout)
        self.assertIn("PRESENT known (known.csv)", completed.stdout)
        self.assertIn("MISSING missing_required (missing_required.csv)", completed.stdout)
        self.assertIn(f"{root / 'root_only.csv'} [tradingview_pine_export_candidate]", completed.stdout)
        self.assertIn(str(pine_dir / "lcv6.pine"), completed.stdout)

    def test_audit_fixtures_json_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            files = root / "Files"
            pine_dir = root / "PineScript" / "Indicators"
            external_dir = root / "Indicators" / "LorentzianClassification"
            files.mkdir(parents=True)
            pine_dir.mkdir(parents=True)
            external_dir.mkdir(parents=True)
            (files / "known.csv").write_text("time,open,high,low,close\n")
            (root / "candidate.csv").write_text(
                "time,open,high,low,close,Kernel Regression Estimate,Prediction,Direction,Buy,Sell\n"
            )
            (pine_dir / "lcv6.pine").write_text('indicator("fixture")\n')
            (external_dir / "LorentzianClassification.external_src").write_text("// indicator fixture\n")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "pine_sources": [
                            {
                                "name": "lcv6",
                                "path": "PineScript/Indicators/lcv6.pine",
                                "role": "canonical",
                            }
                        ],
                        "external_sources": [
                            {
                                "name": "LorentzianClassification",
                                "path": "Indicators/LorentzianClassification/LorentzianClassification.external_src",
                                "role": "canonical external",
                            }
                        ],
                        "fixtures": [
                            {
                                "name": "known",
                                "filename": "known.csv",
                                "settings": {},
                            }
                        ],
                        "required_uncovered_fixture_cases": [
                            {
                                "name": "missing_required",
                                "filename": "missing_required.csv",
                                "python_smoke_fixture": "known.csv",
                                "proves": ["audit JSON required export inventory"],
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "audit-fixtures",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["summary"]["tracked_present"], 1)
        self.assertEqual(payload["summary"]["required_missing"], 1)
        self.assertEqual(payload["summary"]["untracked_pine_export_candidates"], 1)
        self.assertEqual(payload["summary"]["unexpected_pine_export_candidates"], 1)
        self.assertEqual(payload["manifest_pine_sources"][0]["name"], "lcv6")
        self.assertFalse(payload["manifest_pine_sources"][0]["debug_markers"])
        self.assertEqual(payload["summary"]["external_sources_present"], 1)
        self.assertEqual(payload["summary"]["external_sources_invalid"], 0)
        self.assertEqual(payload["manifest_external_sources"][0]["name"], "LorentzianClassification")
        self.assertEqual(
            payload["untracked_csv_candidates"][0]["classification"],
            "tradingview_pine_export_candidate",
        )

    def test_audit_fixtures_catalogs_ignored_csv_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            root.mkdir()
            (root / "tdann.csv").write_text(
                "time,open,high,low,close,Kernel Regression Estimate,Prediction,Direction,Buy,Sell\n"
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [],
                        "ignored_csv_candidates": [
                            {
                                "filename": "tdann.csv",
                                "reason": "separate TDANN Pine variant, outside lcv6 parity",
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "audit-fixtures",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["summary"]["ignored_csv_candidates"], 1)
        self.assertEqual(payload["summary"]["untracked_csv_candidates"], 0)
        self.assertEqual(payload["summary"]["unexpected_pine_export_candidates"], 0)
        self.assertEqual(payload["ignored_csv_candidates"][0]["filename"], "tdann.csv")
        self.assertEqual(
            payload["ignored_csv_candidates"][0]["classification"],
            "tradingview_pine_export_candidate",
        )
        self.assertEqual(payload["untracked_csv_candidates"], [])

    def test_readiness_fails_on_unexpected_pine_export_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            root.mkdir()
            (root / "surprise.csv").write_text(
                "time,open,high,low,close,Kernel Regression Estimate,Prediction,Direction,Buy,Sell\n"
            )
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps({"fixtures": []}))

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "readiness",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["summary"]["unexpected_pine_export_candidates"], 1)
        self.assertEqual(payload["unexpected_pine_export_candidates"][0]["filename"], "surprise.csv")
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_manifest_rejects_ignored_csv_overlap_with_tracked_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [{"name": "tracked", "filename": "Evidence.csv", "settings": {}}],
                        "ignored_csv_candidates": [
                            {"filename": "evidence.csv", "reason": "duplicate evidence route"}
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "validate-fixtures",
                    "--fixture-dir",
                    str(Path(tmp)),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "ignored_csv_candidates overlap tracked, required, or external parity report filenames ignoring case: "
            "Evidence.csv / evidence.csv",
            completed.stdout,
        )
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_audit_fixtures_classifies_directory_csv_candidates_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external"
            files = root / "Files"
            files.mkdir(parents=True)
            (files / "known.csv").write_text("time,open,high,low,close\n")
            csv_directory = root / "not_an_export.csv"
            csv_directory.mkdir()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fixtures": [
                            {
                                "name": "known",
                                "filename": "known.csv",
                                "settings": {},
                            }
                        ],
                    }
                )
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lorentzian_classification",
                    "audit-fixtures",
                    "--fixture-dir",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--json",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(PYTHON_PORT)},
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        candidates = {
            row["filename"]: row["classification"]
            for row in payload["untracked_csv_candidates"]
        }
        self.assertEqual(candidates["not_an_export.csv"], "not_a_file")

    def test_csv_candidate_classifier_identifies_fixture_evidence_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = {
                "pine.csv": (
                    "time,open,high,low,close,F1_RSI,Kernel Regression Estimate,Prediction,Direction,Buy,Sell\n",
                    "tradingview_pine_export_candidate",
                ),
                "comparison.csv": (
                    "time,close,tv_prediction,external_prediction,pred_match,tv_kernel,external_kernel,kernel_diff\n",
                    "external_parity_comparison",
                ),
                "smoke.csv": (
                    "status,symbol,timeframe,bars,time,close,kernel,direction,prediction,error\n",
                    "indicator_smoke_report",
                ),
                "external_runtime_export.csv": (
                    "time,unix,open,high,low,close,buy,sell,exitbuy,exitsell,kernel,direction,prediction\n",
                    "external_runtime_generated_indicator_export",
                ),
                "ticks.csv": (
                    "2026.02.24\t03:00:00.008\t64656.01\t64656.02\t64656.01\t0.123780\n",
                    "tick_import_data",
                ),
                "ohlc.csv": (
                    "time,open,high,low,close\n",
                    "ohlc_price_csv",
                ),
            }
            for filename, (header, expected) in cases.items():
                path = tmp_path / filename
                path.write_text(header)
                self.assertEqual(classify_csv_candidate(path), expected, filename)

            directory_path = tmp_path / "directory.csv"
            directory_path.mkdir()
            self.assertEqual(classify_csv_candidate(directory_path), "not_a_file")
            self.assertEqual(classify_csv_candidate(tmp_path / "missing.csv"), "missing_csv")

    def test_reader_accepts_slot_feature_headers_and_false_signal_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "custom_export.csv"
            fixture.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,F1_CCI,F2_RSI,F3_WT,F4_ADX,F5_CCI,"
                        "Kernel Regression Estimate,Prediction,Direction,Buy,Sell,StopBuy,StopSell",
                        "2026-01-01,1.1,1.2,1.0,1.15,0.1,0.2,0.3,0.4,0.5,1.12,-1,-1,0,false,na,",
                        "2026-01-02,1.2,1.3,1.1,1.25,0.6,0.7,0.8,0.9,1.0,1.18,1,1,1,true,1,true",
                    ]
                )
            )

            rows, price_scale = read_tradingview_csv(fixture)

        self.assertEqual(price_scale, 100.0)
        self.assertEqual(rows[0].f1, 0.1)
        self.assertEqual(rows[0].f5, 0.5)
        self.assertFalse(rows[0].buy)
        self.assertFalse(rows[0].sell)
        self.assertFalse(rows[0].exit_buy)
        self.assertFalse(rows[0].exit_sell)
        self.assertTrue(rows[1].buy)
        self.assertTrue(rows[1].sell)
        self.assertTrue(rows[1].exit_buy)
        self.assertTrue(rows[1].exit_sell)

    def test_reader_parses_optional_alert_and_trade_stat_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "instrumented_export.csv"
            fixture.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,Prediction,Direction,Buy,Sell,StopBuy,StopSell,"
                        "Open Long Alert,Close Long Alert,Open Short Alert,Close Short Alert,"
                        "Open Position Alert,Close Position Alert,Kernel Bullish Alert,Kernel Bearish Alert,"
                        "Kernel Plot Color,Prediction Label,Prediction Label Y,Prediction Label Color,Bar Color,"
                        "Trade Stats Visible,Trade Stats Header,Total Wins,Total Losses,Total Early Signal Flips,"
                        "Total Trades,Win Loss Ratio,Table WL Ratio,Win Rate",
                        "2026-01-01,1.1,1.2,1.0,1.15,0,0,,,,,1,0,true,false,1,0,false,true,"
                        "#009988@20,0,1.25,#787b86@25,,1,Trade Stats,2,1,3,3,0.6666666667,2,0.6666666667",
                    ]
                )
            )

            [row], _price_scale = read_tradingview_csv(fixture)

        self.assertTrue(row.open_long_alert)
        self.assertFalse(row.close_long_alert)
        self.assertTrue(row.open_short_alert)
        self.assertFalse(row.close_short_alert)
        self.assertTrue(row.open_position_alert)
        self.assertFalse(row.close_position_alert)
        self.assertFalse(row.kernel_bullish_alert)
        self.assertTrue(row.kernel_bearish_alert)
        self.assertEqual(row.kernel_plot_color, "#009988@20")
        self.assertEqual(row.prediction_label, "0")
        self.assertEqual(row.prediction_label_y, 1.25)
        self.assertEqual(row.prediction_label_color, "#787b86@25")
        self.assertEqual(row.bar_color, "")
        self.assertTrue(row.trade_stats_visible)
        self.assertEqual(row.trade_stats_header, "Trade Stats")
        self.assertEqual(row.total_wins, 2)
        self.assertEqual(row.total_losses, 1)
        self.assertEqual(row.total_early_signal_flips, 3)
        self.assertEqual(row.total_trades, 3)
        self.assertAlmostEqual(row.win_loss_ratio or 0.0, 0.6666666667)
        self.assertEqual(row.table_wl_ratio, 2)
        self.assertAlmostEqual(row.win_rate or 0.0, 0.6666666667)

    def test_reader_decodes_numeric_helper_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "instrumented_export.csv"
            fixture.write_text(
                "\n".join(
                    [
                        "time,open,high,low,close,Prediction,Direction,Buy,Sell,StopBuy,StopSell,"
                        "Kernel Plot Color,Prediction Label,Prediction Label Color,Bar Color,Trade Stats Header",
                        "2026-01-01,1.1,1.2,1.0,1.15,0,0,,,,,120,3.0,225,-150,1",
                    ]
                )
            )

            [row], _price_scale = read_tradingview_csv(fixture)

        self.assertEqual(row.kernel_plot_color, "#009988@20")
        self.assertEqual(row.prediction_label, "3")
        self.assertEqual(row.prediction_label_color, "#787b86@25")
        self.assertEqual(row.bar_color, "#CC3311@50")
        self.assertEqual(row.trade_stats_header, "\U0001f4c8 Trade Stats")

    def test_parity_summary_compares_optional_alert_and_trade_stat_columns(self) -> None:
        bar = Bar("2026-01-01", 1.1, 1.2, 1.0, 1.15)
        tv = TvRow(
            bar=bar,
            open_long_alert=True,
            close_long_alert=False,
            open_short_alert=False,
            close_short_alert=True,
            open_position_alert=True,
            close_position_alert=False,
            kernel_bullish_alert=True,
            kernel_bearish_alert=False,
            kernel_plot_color="#009988@20",
            prediction_label="0",
            prediction_label_y=1.25,
            prediction_label_color="#787b86@25",
            bar_color="#787b86@50",
            trade_stats_visible=True,
            trade_stats_header="Trade Stats",
            total_wins=2,
            total_losses=1,
            total_early_signal_flips=1,
            total_trades=3,
            win_loss_ratio=2 / 3,
            table_wl_ratio=2,
            win_rate=2 / 3,
        )
        py = ResultRow(
            bar=bar,
            f1=float("nan"),
            f2=float("nan"),
            f3=float("nan"),
            f4=float("nan"),
            f5=float("nan"),
            kernel=float("nan"),
            prediction=0,
            direction=0,
            buy=False,
            sell=False,
            exit_buy=False,
            exit_sell=False,
            stop_buy=False,
            stop_sell=False,
            backtest_stream=0,
            open_long_alert=True,
            close_long_alert=False,
            open_short_alert=False,
            close_short_alert=True,
            open_position_alert=True,
            close_position_alert=False,
            kernel_bullish_alert=True,
            kernel_bearish_alert=False,
            kernel_plot_color="#009988@20",
            prediction_label="0",
            prediction_label_y=1.25,
            prediction_label_color="#787b86@25",
            bar_color="#787b86@50",
            trade_stats_header="Trade Stats",
            total_wins=2,
            total_losses=1,
            total_early_signal_flips=1,
            total_trades=3,
            win_loss_ratio=2 / 3,
            table_wl_ratio=2,
            win_rate=2 / 3,
        )

        summary, mismatches = parity_summary([tv], [py], 1e-9)
        self.assertTrue(summary["pass"])
        self.assertFalse(mismatches)
        self.assertEqual(summary["optional_compared"]["open_long_alert"], 1)
        self.assertEqual(summary["optional_compared"]["open_position_alert"], 1)
        self.assertEqual(summary["optional_compared"]["kernel_plot_color"], 1)
        self.assertEqual(summary["optional_compared"]["trade_stats_visible"], 1)
        self.assertEqual(summary["optional_compared"]["trade_stats_header"], 1)
        self.assertEqual(summary["optional_compared"]["table_wl_ratio"], 1)
        self.assertEqual(summary["optional_compared"]["win_rate"], 1)

        bad_summary, bad_mismatches = parity_summary(
            [tv],
            [replace(py, total_trades=4, bar_color="#CC3311@50", trade_stats_visible=False, table_wl_ratio=3)],
            1e-9,
        )
        self.assertFalse(bad_summary["pass"])
        self.assertEqual(bad_summary["optional_mismatches"]["total_trades"], 1)
        self.assertEqual(bad_summary["optional_mismatches"]["bar_color"], 1)
        self.assertEqual(bad_summary["optional_mismatches"]["trade_stats_visible"], 1)
        self.assertEqual(bad_summary["optional_mismatches"]["table_wl_ratio"], 1)
        self.assertEqual(len(bad_mismatches), 1)
        self.assertEqual(
            bad_mismatches[0]["mismatch_reasons"],
            "bar_color;trade_stats_visible;total_trades;table_wl_ratio",
        )

    def test_parity_summary_fails_on_row_count_and_missing_numeric_outputs(self) -> None:
        bar = Bar("2026-01-01", 1.1, 1.2, 1.0, 1.15)
        tv = TvRow(bar=bar, f1=0.25, prediction=0, direction=0)
        py = ResultRow(
            bar=bar,
            f1=float("nan"),
            f2=float("nan"),
            f3=float("nan"),
            f4=float("nan"),
            f5=float("nan"),
            kernel=float("nan"),
            prediction=0,
            direction=0,
            buy=False,
            sell=False,
            exit_buy=False,
            exit_sell=False,
            stop_buy=False,
            stop_sell=False,
            backtest_stream=0,
            open_long_alert=False,
            close_long_alert=False,
            open_short_alert=False,
            close_short_alert=False,
            open_position_alert=False,
            close_position_alert=False,
            kernel_bullish_alert=False,
            kernel_bearish_alert=False,
            kernel_plot_color="",
            prediction_label="0",
            prediction_label_y=0.0,
            prediction_label_color="",
            bar_color="",
            trade_stats_header="",
            total_wins=0,
            total_losses=0,
            total_early_signal_flips=0,
            total_trades=0,
            win_loss_ratio=float("nan"),
            table_wl_ratio=float("nan"),
            win_rate=float("nan"),
        )

        missing_numeric, mismatches = parity_summary([tv], [py], 1e-9)
        self.assertFalse(missing_numeric["pass"])
        self.assertEqual(missing_numeric["compared"]["f1"], 1)
        self.assertEqual(missing_numeric["numeric_mismatches"]["f1"], 1)
        self.assertEqual(mismatches[0]["mismatch_reasons"], "f1")

        row_count, _mismatches = parity_summary([tv, tv], [replace(py, f1=0.25)], 1e-9)
        self.assertFalse(row_count["pass"])
        self.assertTrue(row_count["row_count_mismatch"])
        self.assertEqual(row_count["rows"], 2)
        self.assertEqual(row_count["python_rows"], 1)

    def test_parity_summary_reports_warmup_numeric_mismatch_details(self) -> None:
        warmup = Bar("2026-01-01", 1.1, 1.2, 1.0, 1.15)
        active = Bar("2026-01-02", 1.2, 1.3, 1.1, 1.25)
        tv_rows = [
            TvRow(bar=warmup, f1=0.25),
            TvRow(bar=active, f1=0.5),
            TvRow(bar=active, f1=0.75),
        ]
        py_template = ResultRow(
            bar=warmup,
            f1=0.0,
            f2=float("nan"),
            f3=float("nan"),
            f4=float("nan"),
            f5=float("nan"),
            kernel=float("nan"),
            prediction=0,
            direction=0,
            buy=False,
            sell=False,
            exit_buy=False,
            exit_sell=False,
            stop_buy=False,
            stop_sell=False,
            backtest_stream=0,
            open_long_alert=False,
            close_long_alert=False,
            open_short_alert=False,
            close_short_alert=False,
            open_position_alert=False,
            close_position_alert=False,
            kernel_bullish_alert=False,
            kernel_bearish_alert=False,
            kernel_plot_color="",
            prediction_label="0",
            prediction_label_y=0.0,
            prediction_label_color="",
            bar_color="",
            trade_stats_header="",
            total_wins=0,
            total_losses=0,
            total_early_signal_flips=0,
            total_trades=0,
            win_loss_ratio=float("nan"),
            table_wl_ratio=float("nan"),
            win_rate=float("nan"),
        )
        py_rows = [
            py_template,
            replace(py_template, bar=active, f1=0.5),
            replace(py_template, bar=active, f1=0.75),
        ]

        summary, mismatches = parity_summary(tv_rows, py_rows, 1e-9, Settings(max_bars_back=1))

        self.assertFalse(summary["pass"])
        self.assertEqual(summary["max_bars_back_index"], 1)
        self.assertEqual(summary["numeric_mismatches"]["f1"], 1)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["time"], "2026-01-01")
        self.assertEqual(mismatches[0]["mismatch_reasons"], "f1")

    def test_manifest_can_encode_non_default_pine_export_settings(self) -> None:
        manifest = {
            "fixtures": [
                {
                    "name": "non_default_export",
                    "filename": "non_default.csv",
                    "tolerance": 1e-8,
                    "settings": {
                        "include_full_history": True,
                        "source": "hlc3",
                        "feature_count": 3,
                        "use_adx_filter": True,
                        "use_ema_filter": True,
                        "use_sma_filter": True,
                        "use_kernel_smoothing": True,
                        "use_dynamic_exits": True,
                        "show_exits": True,
                        "use_worst_case": True,
                        "show_trade_stats": False,
                        "kernel_h": 12,
                        "kernel_r": 4.5,
                        "kernel_x": 20,
                        "kernel_lag": 1,
                        "f1": "RSI:21:2",
                        "f2": ["WT", 8, 10],
                        "f3": "CCI:34:2",
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            [spec] = load_parity_manifest(manifest_path)

        self.assertEqual(spec.filename, "non_default.csv")
        self.assertEqual(spec.tolerance, 1e-8)
        self.assertEqual(spec.settings.source, "hlc3")
        self.assertEqual(spec.settings.feature_count, 3)
        self.assertTrue(spec.settings.use_adx_filter)
        self.assertTrue(spec.settings.use_ema_filter)
        self.assertTrue(spec.settings.use_sma_filter)
        self.assertTrue(spec.settings.use_kernel_smoothing)
        self.assertTrue(spec.settings.use_dynamic_exits)
        self.assertTrue(spec.settings.show_exits)
        self.assertTrue(spec.settings.use_worst_case)
        self.assertFalse(spec.settings.show_trade_stats)
        self.assertEqual(spec.settings.kernel_h, 12)
        self.assertEqual(spec.settings.kernel_r, 4.5)
        self.assertEqual(spec.settings.kernel_x, 20)
        self.assertEqual(spec.settings.kernel_lag, 1)
        self.assertEqual(spec.settings.f1, ("RSI", 21, 2))
        self.assertEqual(spec.settings.f2, ("WT", 8, 10))
        self.assertEqual(spec.settings.f3, ("CCI", 34, 2))

    def test_every_local_pine_input_has_a_cli_setting(self) -> None:
        if not PINE_LCV6.exists():
            raise unittest.SkipTest(f"Pine source not found: {PINE_LCV6}")
        source = PINE_LCV6.read_text()
        input_to_setting = {
            "Source": "source",
            "Neighbors Count": "neighbors_count",
            "Max Bars Back": "max_bars_back",
            "Feature Count": "feature_count",
            "Color Compression": "color_compression",
            "Show Default Exits": "show_exits",
            "Use Dynamic Exits": "use_dynamic_exits",
            "showTradeStats": "show_trade_stats",
            "useWorstCase": "use_worst_case",
            "includeFullHistory": "include_full_history",
            "Use Volatility Filter": "use_volatility_filter",
            "Use Regime Filter": "use_regime_filter",
            "Use ADX Filter": "use_adx_filter",
            "Threshold|regime": "regime_threshold",
            "Threshold|adx": "adx_threshold",
            "f1_string": "f1",
            "f1_paramA": "f1",
            "f1_paramB": "f1",
            "f2_string": "f2",
            "f2_paramA": "f2",
            "f2_paramB": "f2",
            "f3_string": "f3",
            "f3_paramA": "f3",
            "f3_paramB": "f3",
            "f4_string": "f4",
            "f4_paramA": "f4",
            "f4_paramB": "f4",
            "f5_string": "f5",
            "f5_paramA": "f5",
            "f5_paramB": "f5",
            "useEmaFilter": "use_ema_filter",
            "emaPeriod": "ema_period",
            "useSmaFilter": "use_sma_filter",
            "smaPeriod": "sma_period",
            "useKernelFilter": "use_kernel_filter",
            "showKernelEstimate": "show_kernel_estimate",
            "useKernelSmoothing": "use_kernel_smoothing",
            "h": "kernel_h",
            "r": "kernel_r",
            "x": "kernel_x",
            "lag": "kernel_lag",
            "showBarColors": "show_bar_colors",
            "showBarPredictions": "show_bar_predictions",
            "useAtrOffset": "use_atr_offset",
            "barPredictionsOffset": "bar_predictions_offset",
            "useConfidenceGradient": "use_confidence_gradient",
        }
        pine_inputs = pine_input_identifiers(source)
        self.assertEqual(pine_inputs - set(input_to_setting), set())
        self.assertEqual(set(input_to_setting) - pine_inputs, set())
        self.assertEqual(set(input_to_setting.values()), {field.name for field in fields(Settings)})

    def test_cli_defaults_match_pine_inputs(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "input.csv", "--output", "output.csv"])
        settings = settings_from_args(args)
        self.assertEqual(settings.source, "close")
        self.assertEqual(settings.neighbors_count, 8)
        self.assertEqual(settings.max_bars_back, 2000)
        self.assertEqual(settings.feature_count, 5)
        self.assertEqual(settings.color_compression, 1)
        self.assertFalse(settings.include_full_history)
        self.assertTrue(settings.use_volatility_filter)
        self.assertTrue(settings.use_regime_filter)
        self.assertFalse(settings.use_adx_filter)
        self.assertEqual(settings.regime_threshold, -0.1)
        self.assertEqual(settings.adx_threshold, 20)
        self.assertFalse(settings.use_ema_filter)
        self.assertEqual(settings.ema_period, 200)
        self.assertFalse(settings.use_sma_filter)
        self.assertEqual(settings.sma_period, 200)
        self.assertTrue(settings.use_kernel_filter)
        self.assertFalse(settings.use_kernel_smoothing)
        self.assertEqual(settings.kernel_h, 8)
        self.assertEqual(settings.kernel_r, 8.0)
        self.assertEqual(settings.kernel_x, 25)
        self.assertEqual(settings.kernel_lag, 2)
        self.assertTrue(settings.show_kernel_estimate)
        self.assertTrue(settings.show_bar_colors)
        self.assertTrue(settings.show_bar_predictions)
        self.assertTrue(settings.use_atr_offset)
        self.assertEqual(settings.bar_predictions_offset, 0)
        self.assertTrue(settings.use_confidence_gradient)
        self.assertTrue(settings.show_trade_stats)
        self.assertFalse(settings.show_exits)
        self.assertFalse(settings.use_dynamic_exits)
        self.assertFalse(settings.use_worst_case)
        self.assertEqual(settings.f1, ("RSI", 14, 1))
        self.assertEqual(settings.f2, ("WT", 10, 11))
        self.assertEqual(settings.f3, ("CCI", 20, 1))
        self.assertEqual(settings.f4, ("ADX", 20, 2))
        self.assertEqual(settings.f5, ("RSI", 9, 1))

    def test_cli_defaults_are_backed_by_local_pine_source(self) -> None:
        if not PINE_LCV6.exists():
            raise unittest.SkipTest(f"Pine source not found: {PINE_LCV6}")
        source = PINE_LCV6.read_text()
        settings = settings_from_args(build_parser().parse_args(["run", "input.csv", "--output", "output.csv"]))

        self.assertEqual(settings.source, pine_defval_for_title(source, "Source"))
        self.assertEqual(settings.neighbors_count, pine_defval_for_title(source, "Neighbors Count"))
        self.assertEqual(settings.max_bars_back, pine_defval_for_title(source, "Max Bars Back"))
        self.assertEqual(settings.feature_count, pine_defval_for_title(source, "Feature Count"))
        self.assertEqual(settings.color_compression, pine_defval_for_title(source, "Color Compression"))
        self.assertEqual(settings.show_exits, pine_defval_for_title(source, "Show Default Exits"))
        self.assertEqual(settings.use_dynamic_exits, pine_defval_for_title(source, "Use Dynamic Exits"))
        self.assertEqual(settings.use_worst_case, pine_positional_default(source, "useWorstCase"))
        self.assertEqual(settings.include_full_history, pine_defval_for_title(source, "Include Full History"))

        self.assertEqual(settings.use_volatility_filter, pine_defval_for_title(source, "Use Volatility Filter"))
        self.assertEqual(settings.use_regime_filter, pine_defval_for_title(source, "Use Regime Filter"))
        self.assertEqual(settings.use_adx_filter, pine_defval_for_title(source, "Use ADX Filter"))
        self.assertEqual(settings.regime_threshold, pine_defval_for_inline(source, "Threshold", "regime"))
        self.assertEqual(settings.adx_threshold, pine_defval_for_inline(source, "Threshold", "adx"))
        self.assertEqual(settings.use_ema_filter, pine_defval_for_title(source, "Use EMA Filter"))
        self.assertEqual(settings.ema_period, pine_defval_for_variable(source, "emaPeriod"))
        self.assertEqual(settings.use_sma_filter, pine_defval_for_title(source, "Use SMA Filter"))
        self.assertEqual(settings.sma_period, pine_defval_for_variable(source, "smaPeriod"))

        self.assertEqual(settings.use_kernel_filter, pine_positional_default(source, "useKernelFilter"))
        self.assertEqual(settings.show_kernel_estimate, pine_positional_default(source, "showKernelEstimate"))
        self.assertEqual(settings.use_kernel_smoothing, pine_positional_default(source, "useKernelSmoothing"))
        self.assertEqual(settings.kernel_h, pine_positional_default(source, "h"))
        self.assertEqual(settings.kernel_r, pine_positional_default(source, "r"))
        self.assertEqual(settings.kernel_x, pine_positional_default(source, "x"))
        self.assertEqual(settings.kernel_lag, pine_positional_default(source, "lag"))
        self.assertEqual(settings.show_bar_colors, pine_positional_default(source, "showBarColors"))
        self.assertEqual(settings.show_bar_predictions, pine_defval_for_variable(source, "showBarPredictions"))
        self.assertEqual(settings.use_atr_offset, pine_defval_for_variable(source, "useAtrOffset"))
        self.assertEqual(settings.bar_predictions_offset, pine_positional_default(source, "barPredictionsOffset"))
        self.assertEqual(settings.use_confidence_gradient, pine_defval_for_variable(source, "useConfidenceGradient"))
        self.assertEqual(settings.show_trade_stats, pine_positional_default(source, "showTradeStats"))

        for index, feature in enumerate([settings.f1, settings.f2, settings.f3, settings.f4, settings.f5], start=1):
            self.assertEqual(feature[0], pine_defval_for_title(source, f"Feature {index}"))
            self.assertEqual(feature[1], pine_defval_for_variable(source, f"f{index}_paramA"))
            self.assertEqual(feature[2], pine_defval_for_variable(source, f"f{index}_paramB"))

    def test_python_core_covers_pine_business_logic_surfaces(self) -> None:
        required_sources = [PINE_LCV6, PINE_ML_EXTENSIONS, PINE_KERNEL_FUNCTIONS]
        missing_sources = [path for path in required_sources if not path.exists()]
        if missing_sources:
            raise unittest.SkipTest(f"Pine source not found: {missing_sources[0]}")

        lcv6_source = PINE_LCV6.read_text()
        ml_source = PINE_ML_EXTENSIONS.read_text()
        kernel_source = PINE_KERNEL_FUNCTIONS.read_text()
        python_source = PYTHON_CORE.read_text()

        surface_map = {
            "export n_rsi": ("ml.normalized RSI feature", ml_source, "def calc_normalized_rsi"),
            "export n_cci": ("ml.normalized CCI feature", ml_source, "def calc_normalized_cci"),
            "export n_wt": ("ml.normalized WaveTrend feature", ml_source, "def calc_wavetrend"),
            "export n_adx": ("ml.normalized ADX feature", ml_source, "def calc_adx"),
            "export regime_filter": ("regime filter", ml_source, "def calc_regime_filter"),
            "export filter_adx": ("ADX filter", ml_source, "def calc_raw_adx_filter"),
            "export filter_volatility": ("volatility ATR filter", ml_source, "vol_atr1"),
            "export analyzePerformance": ("trade-stat backtest analysis", ml_source, "total_early_flips"),
            "export rationalQuadratic": ("rational quadratic kernel", kernel_source, "def kernel_rational_quadratic"),
            "export gaussian": ("Gaussian kernel", kernel_source, "def kernel_gaussian"),
            "get_lorentzian_distance": ("Lorentzian distance", lcv6_source, "def distance"),
            "prediction :=": ("ANN prediction accumulation", lcv6_source, "prediction = ann.run"),
            "filter_all": ("combined volatility/regime/ADX filter", lcv6_source, "filter_all ="),
            "startLongTrade": ("long entry condition", lcv6_source, "start_long ="),
            "startShortTrade": ("short entry condition", lcv6_source, "start_short ="),
            "endLongTradeDynamic": ("dynamic long exit", lcv6_source, "end_long_dynamic ="),
            "endShortTradeDynamic": ("dynamic short exit", lcv6_source, "end_short_dynamic ="),
            "endLongTradeStrict": ("strict long exit", lcv6_source, "end_long_strict ="),
            "endShortTradeStrict": ("strict short exit", lcv6_source, "end_short_strict ="),
            "backTestStream": ("backtest stream", lcv6_source, "backtest_stream ="),
            "showTradeStats": ("trade-stat display gate", lcv6_source, "show_trade_stats"),
            "totalWins": ("total wins output", lcv6_source, "total_wins"),
            "winLossRatio": ("win/loss ratio output", lcv6_source, "win_loss_ratio"),
            "winRate": ("win-rate output", lcv6_source, "win_rate"),
        }

        for pine_token, (description, pine_source, python_token) in surface_map.items():
            self.assertIn(pine_token, pine_source, description)
            self.assertIn(python_token, python_source, description)

        for item in PINE_EXPORT_SERIES:
            expression = str(item["pine_expression"])
            if item["export_mode"] == "chart_export_builtin":
                continue
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression):
                if token in {"and", "direction", "na", "or", "str", "tostring"}:
                    continue
                self.assertIn(token, lcv6_source, f"{item['column']} export expression")

    def test_export_series_tracks_pine_signal_plots_and_alerts(self) -> None:
        if not PINE_LCV6.exists():
            raise unittest.SkipTest(f"Pine source not found: {PINE_LCV6}")

        source = PINE_LCV6.read_text()
        plot_records = pine_call_records(source, "plot")
        plotshape_records = pine_call_records(source, "plotshape")
        alert_records = pine_call_records(source, "alertcondition")
        export_by_column = {str(item["column"]): str(item["pine_expression"]) for item in PINE_EXPORT_SERIES}

        plot_columns = {
            "Kernel Regression Estimate": "Kernel Regression Estimate",
            "Backtest Stream": "Backtest Stream",
        }
        for pine_title, export_column in plot_columns.items():
            self.assertIn(pine_title, plot_records)
            self.assertEqual(plot_records[pine_title], export_by_column[export_column])

        plotshape_columns = {
            "Buy": "Buy",
            "Sell": "Sell",
            "StopBuy": "StopBuy",
            "StopSell": "StopSell",
        }
        self.assertEqual(set(plotshape_records), set(plotshape_columns))
        for pine_title, export_column in plotshape_columns.items():
            self.assertEqual(
                pine_condition_head(plotshape_records[pine_title]),
                pine_condition_head(export_by_column[export_column]),
                pine_title,
            )

        alert_columns = {
            "Open Long ▲": "Open Long Alert",
            "Close Long ▲": "Close Long Alert",
            "Open Short ▼": "Open Short Alert",
            "Close Short ▼": "Close Short Alert",
            "Open Position ▲▼": "Open Position Alert",
            "Close Position ▲▼": "Close Position Alert",
            "Kernel Bullish Color Change": "Kernel Bullish Alert",
            "Kernel Bearish Color Change": "Kernel Bearish Alert",
        }
        self.assertEqual(set(alert_records), set(alert_columns))
        for pine_title, export_column in alert_columns.items():
            self.assertEqual(
                pine_condition_head(alert_records[pine_title]),
                pine_condition_head(export_by_column[export_column]),
                pine_title,
            )

    def test_backtest_stream_and_table_exports_track_pine_formulas(self) -> None:
        if not PINE_LCV6.exists():
            raise unittest.SkipTest(f"Pine source not found: {PINE_LCV6}")

        source = PINE_LCV6.read_text()
        export_by_column = {str(item["column"]): str(item["pine_expression"]) for item in PINE_EXPORT_SERIES}
        self.assertEqual(
            pine_switch_cases(source, "backTestStream"),
            {
                "startLongTrade": "1",
                "endLongTrade": "2",
                "startShortTrade": "-1",
                "endShortTrade": "-2",
            },
        )
        self.assertEqual(export_by_column["Backtest Stream"], "backTestStream")

        self.assertIn("str.tostring(totalWins / totalTrades, '#.#%')", source)
        self.assertIn("str.tostring(totalWins / totalLosses, '0.00')", source)
        self.assertEqual(export_by_column["Win Loss Ratio"], "winLossRatio")
        self.assertEqual(export_by_column["Table WL Ratio"], "totalWins / totalLosses")
        self.assertEqual(export_by_column["Trade Stats Visible"], "showTradeStats ? 1 : 0")

    def test_cli_rejects_pine_bounded_input_violations(self) -> None:
        parser = build_parser()
        invalid_args = [
            ["--neighbors-count", "0"],
            ["--neighbors-count", "101"],
            ["--feature-count", "1"],
            ["--feature-count", "6"],
            ["--color-compression", "0"],
            ["--color-compression", "11"],
            ["--regime-threshold", "-10.1"],
            ["--regime-threshold", "10.1"],
            ["--adx-threshold", "-1"],
            ["--adx-threshold", "101"],
            ["--ema-period", "0"],
            ["--sma-period", "0"],
            ["--kernel-h", "2"],
            ["--bar-predictions-offset", "-0.1"],
        ]
        for extra_args in invalid_args:
            with self.subTest(extra_args=extra_args):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(["run", "input.csv", "--output", "output.csv", *extra_args])

    def test_manifest_settings_reject_pine_bounded_input_violations(self) -> None:
        invalid_overrides = [
            {"neighbors_count": 0},
            {"neighbors_count": 101},
            {"feature_count": 1},
            {"feature_count": 6},
            {"color_compression": 0},
            {"color_compression": 11},
            {"regime_threshold": -10.1},
            {"regime_threshold": 10.1},
            {"adx_threshold": -1},
            {"adx_threshold": 101},
            {"ema_period": 0},
            {"sma_period": 0},
            {"kernel_h": 2},
            {"bar_predictions_offset": -0.1},
        ]
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    settings_from_mapping(overrides)

    def test_manifest_settings_reject_malformed_setting_types(self) -> None:
        invalid_overrides = [
            {"source": 1},
            {"include_full_history": "true"},
            {"neighbors_count": True},
            {"max_bars_back": 2000.5},
            {"regime_threshold": "0"},
            {"kernel_r": "8.0"},
            {"f1": ["RSI", "14", 1]},
            {"f1": ["RSI", True, 1]},
            {"f1": ["RSI", 14.0, 1]},
            {"f1": ["UNKNOWN", 14, 1]},
        ]
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    settings_from_mapping(overrides)

    def test_display_outputs_match_pine_default_formulas(self) -> None:
        input_path = self.fixture_dir / "pine_coinbase_btcusd_1d_limited_history.csv"
        tv_rows, price_scale = read_tradingview_csv(input_path)
        results = calculate(tv_rows, settings=Settings(), price_scale=price_scale)
        last = results[-1]
        self.assertEqual(last.prediction, -4)
        self.assertEqual(last.prediction_label, "-4")
        self.assertEqual(last.prediction_label_color, "#CC3311@50")
        self.assertEqual(last.bar_color, "#CC3311@50")
        self.assertEqual(last.kernel_plot_color, "#CC3311@20")
        self.assertTrue(last.trade_stats_visible)
        self.assertAlmostEqual(last.prediction_label_y, 76433.52 - (76828.0 - 76433.52), places=8)

        hidden = calculate(
            tv_rows[-5:],
            settings=Settings(show_bar_colors=False, show_bar_predictions=False, show_kernel_estimate=False),
            price_scale=price_scale,
        )[-1]
        self.assertEqual(hidden.prediction_label_color, "")
        self.assertEqual(hidden.bar_color, "")
        self.assertEqual(hidden.kernel_plot_color, "#000000@100")

        stats_hidden = calculate(tv_rows[-5:], settings=Settings(show_trade_stats=False), price_scale=price_scale)[-1]
        self.assertFalse(stats_hidden.trade_stats_visible)
        self.assertEqual(stats_hidden.trade_stats_header, "\U0001f4c8 Trade Stats")

    def test_detect_price_scale_clamps_pathological_precision(self) -> None:
        from lorentzian_classification.core import detect_price_scale

        # A cell with >=19 significant fractional digits must not produce an
        # unbounded scale (mirrors the Rust port's i64-overflow clamp).
        rows = [{"open": "1.0", "high": "1.0", "low": "1.0", "close": "0." + "0" * 18 + "1"}]
        self.assertEqual(detect_price_scale(rows), float(10**18))

    def test_sanitize_text_neutralizes_formula_injection(self) -> None:
        from lorentzian_classification.core import sanitize_text

        for payload in ("=cmd|'/c calc'!A1", "+1+1", "-2+3", "@SUM(A1)", "\t=1", "\rfoo"):
            self.assertEqual(sanitize_text(payload), "'" + payload)
        # Legitimate timestamps and dates are untouched (no-op for real data).
        for benign in ("1417392000", "2024-01-01 00:00", "300", ""):
            self.assertEqual(sanitize_text(benign), benign)
