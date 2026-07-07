import pandas as pd

pred = pd.read_csv("data/processed/walk_forward_full.csv")
real = pd.read_csv("data/processed/master_dataset_full.csv")

pred["date"] = pd.to_datetime(pred["date"])
real["date"] = pd.to_datetime(real["date"])

df = pred.merge(real, on="date")

targets = ["total", "delivery", "digital", "cash"]

for t in targets:

    df[f"{t}_abs_error"] = abs(df[f"{t}_pred"] - df[f"{t}_real"])

    df[f"{t}_pct_error"] = (
        df[f"{t}_abs_error"] /
        df[f"{t}_real"].replace(0, pd.NA)
    ) * 100

df["weekday"] = df["date"].dt.day_name()
df["month_name"] = df["date"].dt.month_name()

print("=" * 70)
print("ERRORI PER GIORNO SETTIMANA")
print("=" * 70)

for t in targets:

    print("\n")
    print(t.upper())

    print(
        df.groupby("weekday")[f"{t}_pct_error"]
        .mean()
        .sort_values(ascending=False)
    )

print("\n")
print("=" * 70)
print("ERRORI PER MESE")
print("=" * 70)

for t in targets:

    print("\n")
    print(t.upper())

    print(
        df.groupby("month_name")[f"{t}_pct_error"]
        .mean()
        .sort_values(ascending=False)
    )

print("\n")
print("=" * 70)
print("TOP 20 ERRORI TOTAL")
print("=" * 70)

cols = [
    "date",
    "total_real",
    "total_pred",
    "total_abs_error",
    "total_pct_error",
    "is_holiday",
    "is_bridge",
    "big_event",
    "avg_temp",
    "rain"
]

print(
    df.sort_values(
        "total_abs_error",
        ascending=False
    )[cols].head(20)
)

df.to_csv(
    "data/processed/error_analysis_full.csv",
    index=False
)

print("\nSalvato:")
print("data/processed/error_analysis_full.csv")
