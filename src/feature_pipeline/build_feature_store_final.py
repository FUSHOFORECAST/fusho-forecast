import os
import pandas as pd

BASE = "data/processed/feature_store.csv"

ENGINES = [
    "reports/external_intelligence/calendar_features.csv",
    "reports/external_intelligence/events_features.csv",
    "reports/external_intelligence/external_features.csv",
]

OUTPUT = "data/processed/feature_store_final.csv"

os.makedirs("data/processed", exist_ok=True)

print("=" * 90)
print("FEATURE STORE FINAL BUILDER")
print("=" * 90)

df = pd.read_csv(BASE)
df["date"] = pd.to_datetime(df["date"])

print(f"Base feature: {len(df.columns)} colonne")
print(f"Righe base: {len(df)}")
print(f"Date base: {df['date'].min()} → {df['date'].max()}")

for engine in ENGINES:
    if not os.path.exists(engine):
        print(f"SKIP: {engine}")
        continue

    ext = pd.read_csv(engine)
    ext["date"] = pd.to_datetime(ext["date"])

    duplicate_cols = [
        col for col in ext.columns
        if col in df.columns and col != "date"
    ]

    ext = ext.drop(columns=duplicate_cols)

    before_cols = len(df.columns)

    df = df.merge(
        ext,
        on="date",
        how="left",
    )

    after_cols = len(df.columns)

    print()
    print(engine)
    print(f"+{after_cols - before_cols} nuove feature")
    print(f"Colonne duplicate escluse: {len(duplicate_cols)}")

# pulizia valori mancanti per feature esterne numeriche
external_cols = [
    col for col in df.columns
    if col.startswith("external_")
    or col.startswith("event_")
    or col.startswith("is_")
    or col.startswith("days_")
]

for col in external_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(0)

df.to_csv(OUTPUT, index=False)

print()
print("=" * 90)
print("FEATURE STORE COMPLETATO")
print("=" * 90)
print("Righe:", len(df))
print("Colonne:", len(df.columns))
print("Date:", df["date"].min(), "→", df["date"].max())

external_added = [
    col for col in df.columns
    if col.startswith("external_")
    or col.startswith("event_")
]

print()
print("External/event feature nel dataset:", len(external_added))
print(external_added)

print()
print("Salvato:", OUTPUT)
