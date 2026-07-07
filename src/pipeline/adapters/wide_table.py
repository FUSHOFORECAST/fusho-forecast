import fnmatch
import os

import pandas as pd

from src.pipeline.adapters.common import (
    add_channel_shares,
    aggregate_channels,
    clean_money,
    slugify,
    trim_trailing_empty_days,
)
from src.pipeline.config import RestaurantConfig

READERS = {
    ".csv": pd.read_csv,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}


def read_table(path) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    reader = READERS.get(ext)
    if reader is None:
        raise ValueError(f"Estensione non supportata per wide_table: {path} (supportate: {sorted(READERS)})")
    return reader(path)


def parse_table(raw_df: pd.DataFrame, source_file: str, config: RestaurantConfig):
    if "date" not in raw_df.columns:
        raise ValueError(
            f"'{source_file}' non ha una colonna 'date': l'adapter wide_table richiede "
            f"una colonna 'date' piu' una colonna per ogni raw label di channel_map."
        )

    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    raw_channels = list(config.channel_map.keys())
    declared_col = config.audit.total_row_label

    rows = []
    audit_rows = []

    for _, record in df.iterrows():
        date = record["date"]

        raw_values = {
            raw_channel: clean_money(record[raw_channel])
            for raw_channel in raw_channels
            if raw_channel in df.columns
        }

        channel_totals = aggregate_channels(raw_values, config.channel_map, config.channels)
        computed_total = sum(channel_totals.values())

        if declared_col in df.columns:
            declared_total = clean_money(record[declared_col])
        else:
            declared_total = computed_total

        row_dict = {
            "date": date,
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "total": computed_total,
            "source_file": source_file,
        }
        row_dict.update(channel_totals)
        row_dict.update({slugify(k): v for k, v in raw_values.items()})

        rows.append(row_dict)

        audit_rows.append({
            "date": date,
            "declared_total": declared_total,
            "computed_total": computed_total,
            "difference": computed_total - declared_total,
            "source_file": source_file,
        })

    return rows, audit_rows


def extract(config: RestaurantConfig, persist: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = config.raw_dir
    files = [
        f for f in os.listdir(raw_dir)
        if fnmatch.fnmatch(f, config.data_source.file_pattern) and not f.startswith("~$")
    ]

    if not files:
        raise FileNotFoundError(f"Nessun file trovato in {raw_dir} (pattern: {config.data_source.file_pattern})")

    all_rows = []
    all_audit = []

    for filename in sorted(files):
        path = raw_dir / filename
        raw_df = read_table(path)

        rows, audit = parse_table(raw_df, filename, config)
        all_rows.extend(rows)
        all_audit.extend(audit)

    df = pd.DataFrame(all_rows)
    audit_df = pd.DataFrame(all_audit)

    if df.empty:
        raise ValueError("Dataset vuoto: non sono stati estratti dati dai file wide_table.")

    df = df.sort_values("date").reset_index(drop=True)
    audit_df = audit_df.sort_values("date").reset_index(drop=True)

    df, audit_df = trim_trailing_empty_days(df, audit_df)

    df = add_channel_shares(df, config.channels)

    if persist:
        os.makedirs(config.processed_dir, exist_ok=True)
        df.to_csv(config.processed_path("master_dataset.csv"), index=False)
        audit_df.to_csv(config.processed_path("audit_totals.csv"), index=False)

    return df, audit_df
