"""Python CLI and library port of Lorentzian Classification."""

from .core import (
    RESULT_FIELDNAMES,
    Bar,
    LorentzianClassification,
    ResultRow,
    TvRow,
    calculate,
    coerce_input_rows,
    read_tradingview_csv,
    result_to_mapping,
    rows_from_records,
    write_result_csv,
)
from .settings import Settings

__version__ = "0.1.0"

__all__ = [
    "Bar",
    "LorentzianClassification",
    "RESULT_FIELDNAMES",
    "ResultRow",
    "Settings",
    "TvRow",
    "__version__",
    "calculate",
    "coerce_input_rows",
    "read_tradingview_csv",
    "result_to_mapping",
    "rows_from_records",
    "write_result_csv",
]
