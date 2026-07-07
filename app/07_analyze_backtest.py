import pandas as pd
import os

INPUT_FILE = "data/processed/backtest_results.csv"
OUTPUT_FILE = "data/processed/backtest_analysis.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

df["dayofweek"] = df["date"].dt.day_name()
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6])

TARGETS = ["total", "delivery", "digital", "cash"]

for target in TARGETS:
    df[f"{target}_abs_error"] = df[f"{target}_error"].abs()
    df[f"{target}_pct_error"] = (
        df[f"{target}_abs_error"] / df[f"{target}_real"].replace(0, pd.NA)
    ) * 100

print("\n=== ERRORI MEDI GENERALI ===")
for target in TARGETS:
    print(
        target.upper(),
        "MAE:",
        round(df[f"{target}_abs_error"].mean(), 2),
        "€ | MAPE:",
        round(df[f"{target}_pct_error"].mean(), 2),
        "%"
    )

print("\n=== ERRORI PER GIORNO SETTIMANA - TOTAL ===")
print(
    df.groupby("dayofweek")["total_pct_error"]
    .mean()
    .sort_values(ascending=False)
)

print("\n=== GIORNI PEGGIORI TOTAL ===")
print(
    df.sort_values("total_abs_error", ascending=False)[
        ["date", "dayofweek", "total_real", "total_pred", "total_error", "total_pct_error"]
    ].head(10)
)

print("\n=== GIORNI PEGGIORI DELIVERY ===")
print(
    df.sort_values("delivery_abs_error", ascending=False)[
        ["date", "dayofweek", "delivery_real", "delivery_pred", "delivery_error", "delivery_pct_error"]
    ].head(10)
)

df.to_csv(OUTPUT_FILE, index=False)
print("\nSalvato:", OUTPUT_FILE)
