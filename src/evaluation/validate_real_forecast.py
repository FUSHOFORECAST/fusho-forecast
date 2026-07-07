import pandas as pd

FORECAST_FILE = "data/processed/forecast_hierarchical_days.csv"
REAL_FILE = "data/processed/june_real_totals.csv"
OUTPUT_FILE = "data/processed/june_real_validation_14_days.csv"

forecast = pd.read_csv(FORECAST_FILE)
real = pd.read_csv(REAL_FILE)

forecast["date"] = pd.to_datetime(forecast["date"])
real["date"] = pd.to_datetime(real["date"])

real = real.dropna(subset=["real_total"]).copy()

df = forecast.merge(real, on="date", how="inner")

df["error"] = df["total_pred"] - df["real_total"]
df["abs_error"] = df["error"].abs()
df["pct_error"] = df["abs_error"] / df["real_total"] * 100

print("=" * 70)
print("VALIDAZIONE REALE 1-14 GIUGNO 2026")
print("=" * 70)

print(df[[
    "date",
    "total_pred",
    "real_total",
    "error",
    "pct_error"
]])

print("\nGIORNI VALIDATI:", len(df))
print("TOTALE PREVISTO:", round(df["total_pred"].sum(), 2))
print("TOTALE REALE:", round(df["real_total"].sum(), 2))
print("ERRORE €:", round(df["total_pred"].sum() - df["real_total"].sum(), 2))
print(
    "ERRORE %:",
    round(
        (df["total_pred"].sum() - df["real_total"].sum())
        / df["real_total"].sum()
        * 100,
        2,
    ),
)
print("MAPE GIORNALIERO:", round(df["pct_error"].mean(), 2), "%")

df.to_csv(OUTPUT_FILE, index=False)

print("\nSalvato:", OUTPUT_FILE)
