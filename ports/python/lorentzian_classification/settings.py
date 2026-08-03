"""Public settings model and validation for Lorentzian Classification."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping


FeatureSpec = tuple[str, int, int]

VALID_SOURCES = ("open", "high", "low", "close", "hl2", "hlc3", "ohlc4")
FEATURE_SLOTS = ("f1", "f2", "f3", "f4", "f5")
FEATURE_TYPES = ("RSI", "WT", "CCI", "ADX")
BOOLEAN_SETTING_FIELDS = {
    "include_full_history",
    "use_volatility_filter",
    "use_regime_filter",
    "use_adx_filter",
    "use_ema_filter",
    "use_sma_filter",
    "use_kernel_filter",
    "use_kernel_smoothing",
    "use_dynamic_exits",
    "show_exits",
    "use_worst_case",
    "show_kernel_estimate",
    "show_bar_colors",
    "show_bar_predictions",
    "use_atr_offset",
    "use_confidence_gradient",
    "show_trade_stats",
}
INTEGER_SETTING_FIELDS = {
    "neighbors_count",
    "max_bars_back",
    "feature_count",
    "color_compression",
    "adx_threshold",
    "ema_period",
    "sma_period",
    "kernel_h",
    "kernel_x",
    "kernel_lag",
}
FLOAT_SETTING_FIELDS = {"regime_threshold", "kernel_r", "bar_predictions_offset"}
PINE_SETTING_BOUNDS: dict[str, tuple[float | int | None, float | int | None]] = {
    "neighbors_count": (1, 100),
    "feature_count": (2, 5),
    "color_compression": (1, 10),
    "regime_threshold": (-10, 10),
    "adx_threshold": (0, 100),
    "ema_period": (1, None),
    "sma_period": (1, None),
    "kernel_h": (3, None),
    "bar_predictions_offset": (0, None),
}


@dataclass(frozen=True)
class Settings:
    """Validated settings shared by the Python library and CLI."""

    source: str = "close"
    neighbors_count: int = 8
    max_bars_back: int = 2000
    feature_count: int = 5
    color_compression: int = 1
    include_full_history: bool = False
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_adx_filter: bool = False
    regime_threshold: float = -0.1
    adx_threshold: int = 20
    use_ema_filter: bool = False
    ema_period: int = 200
    use_sma_filter: bool = False
    sma_period: int = 200
    use_kernel_filter: bool = True
    use_kernel_smoothing: bool = False
    use_dynamic_exits: bool = False
    show_exits: bool = False
    use_worst_case: bool = False
    kernel_h: int = 8
    kernel_r: float = 8.0
    kernel_x: int = 25
    kernel_lag: int = 2
    show_kernel_estimate: bool = True
    show_bar_colors: bool = True
    show_bar_predictions: bool = True
    use_atr_offset: bool = True
    bar_predictions_offset: float = 0.0
    use_confidence_gradient: bool = True
    show_trade_stats: bool = True
    f1: FeatureSpec = ("RSI", 14, 1)
    f2: FeatureSpec = ("WT", 10, 11)
    f3: FeatureSpec = ("CCI", 20, 1)
    f4: FeatureSpec = ("ADX", 20, 2)
    f5: FeatureSpec = ("RSI", 9, 1)

    def __post_init__(self) -> None:
        validate_settings(self)

    @classmethod
    def from_mapping(cls, overrides: Mapping[str, object]) -> Settings:
        """Build validated settings from partial mapping overrides.

        Feature values may use either ``"TYPE:PARAM_A:PARAM_B"`` strings or
        three-item tuples/lists. All omitted values keep their documented
        defaults.
        """

        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(overrides) - field_names)
        if unknown:
            raise ValueError(f"unknown settings keys: {', '.join(unknown)}")

        values = cls().to_mapping()
        for key, value in overrides.items():
            values[key] = coerce_feature(value, key) if key in FEATURE_SLOTS else value
        return cls(**values)

    def to_mapping(self) -> dict[str, object]:
        """Return the effective settings as a plain mapping."""

        return {field.name: getattr(self, field.name) for field in fields(self)}


def is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_feature_string(value: str) -> FeatureSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("feature must have form TYPE:PARAM_A:PARAM_B, e.g. RSI:14:1")
    kind = parts[0].upper()
    if kind not in FEATURE_TYPES:
        raise ValueError("feature type must be one of RSI, WT, CCI, ADX")
    try:
        return kind, int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError("feature params must be integers") from exc


def coerce_feature(value: object, key: str) -> FeatureSpec:
    if isinstance(value, str):
        try:
            return parse_feature_string(value)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
    if isinstance(value, list | tuple) and len(value) == 3:
        kind = str(value[0]).upper()
        if kind not in FEATURE_TYPES:
            raise ValueError(f"{key}: feature type must be one of RSI, WT, CCI, ADX")
        if not is_strict_int(value[1]) or not is_strict_int(value[2]):
            raise ValueError(f"{key}: feature params must be integers")
        return kind, value[1], value[2]
    raise ValueError(f"{key}: feature must be a TYPE:PARAM_A:PARAM_B string or 3-item list")


def validate_settings(settings: Settings) -> Settings:
    """Validate one settings object and return it unchanged."""

    for key in BOOLEAN_SETTING_FIELDS:
        if not isinstance(getattr(settings, key), bool):
            raise ValueError(f"{key}: must be a boolean")

    for key in INTEGER_SETTING_FIELDS:
        if not is_strict_int(getattr(settings, key)):
            raise ValueError(f"{key}: must be an integer")

    for key in FLOAT_SETTING_FIELDS:
        value = getattr(settings, key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key}: must be a number")

    if not isinstance(settings.source, str):
        raise ValueError("source: must be a string")
    if settings.source not in VALID_SOURCES:
        raise ValueError(f"source: must be one of {', '.join(VALID_SOURCES)}")

    for key, (minimum, maximum) in PINE_SETTING_BOUNDS.items():
        value = getattr(settings, key)
        if minimum is not None and value < minimum:
            raise ValueError(f"{key}: must be >= {minimum:g}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{key}: must be <= {maximum:g}")

    for key in FEATURE_SLOTS:
        feature = getattr(settings, key)
        if not isinstance(feature, tuple | list) or len(feature) != 3:
            raise ValueError(f"{key}: feature must be a 3-item tuple")
        kind, param_a, param_b = feature
        if kind not in FEATURE_TYPES:
            raise ValueError(f"{key}: feature type must be one of RSI, WT, CCI, ADX")
        if not is_strict_int(param_a) or not is_strict_int(param_b):
            raise ValueError(f"{key}: feature params must be integers")

    return settings
