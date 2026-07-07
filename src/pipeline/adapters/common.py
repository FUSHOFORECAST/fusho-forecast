import re

import pandas as pd


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def clean_money(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()
    if value == "":
        return 0.0

    value = (
        value.replace("€", "")
        .replace(" ", "")
        .replace(" ", "")
        .replace(".", "")
        .replace(",", ".")
    )

    try:
        return float(value)
    except ValueError:
        return 0.0


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def aggregate_channels(raw_values: dict[str, float], channel_map: dict[str, str], channels: list[str]) -> dict[str, float]:
    channel_totals = {channel: 0.0 for channel in channels}
    for raw_channel, value in raw_values.items():
        channel_totals[channel_map[raw_channel]] += value
    return channel_totals


def add_channel_shares(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    out = df.copy()
    for channel in channels:
        out[f"{channel}_share"] = out[channel] / out["total"].replace(0, pd.NA)
    return out
