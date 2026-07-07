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


def resolve_duplicate_rows(raw_values: dict[str, float], duplicate_row_groups: list[list[str]]) -> dict[str, float]:
    """Alcune fonti registrano la stessa quantita' su piu' righe (es. due
    versioni del contante di cassa). Per ogni gruppo di etichette dichiarate
    equivalenti, prende il valore piu' alto del gruppo (il piu' affidabile in
    caso di disallineamento) e azzera gli altri membri, cosi' la somma per
    canale non conta due volte lo stesso incasso."""
    resolved = dict(raw_values)

    for group in duplicate_row_groups:
        present = [label for label in group if label in resolved]
        if len(present) <= 1:
            continue

        max_value = max(resolved[label] for label in present)
        resolved[present[0]] = max_value
        for label in present[1:]:
            resolved[label] = 0.0

    return resolved


def aggregate_channels(raw_values: dict[str, float], channel_map: dict[str, str], channels: list[str]) -> dict[str, float]:
    channel_totals = {channel: 0.0 for channel in channels}
    for raw_channel, value in raw_values.items():
        channel_totals[channel_map[raw_channel]] += value
    return channel_totals


def trim_trailing_empty_days(df: pd.DataFrame, audit_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rimuove le righe finali (per data) con total=0 da entrambi i dataframe.
    Tipico di un foglio 'vivo' aggiornato giorno per giorno: le colonne dei
    giorni futuri esistono gia' strutturalmente nel mese in corso ma sono
    ancora vuote, non sono dati mancanti da segnalare. Non tocca eventuali
    zeri in mezzo alla serie (potrebbero essere chiusure reali)."""
    nonzero = df[df["total"] != 0]
    if nonzero.empty:
        return df, audit_df

    cutoff_date = nonzero["date"].max()
    return (
        df[df["date"] <= cutoff_date].reset_index(drop=True),
        audit_df[audit_df["date"] <= cutoff_date].reset_index(drop=True),
    )


def add_channel_shares(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    out = df.copy()
    for channel in channels:
        out[f"{channel}_share"] = out[channel] / out["total"].replace(0, pd.NA)
    return out
