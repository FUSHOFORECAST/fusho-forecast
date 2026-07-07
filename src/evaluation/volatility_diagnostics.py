import pandas as pd

INPUT_FILE = "data/processed/june_real_validation_14_days.csv"
OUTPUT_FILE = "reports/volatility_diagnostics.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

df["predicted_change"] = df["total_pred"].diff().abs()
df["real_change"] = df["real_total"].diff().abs()

df["compression_gap"] = df["real_change"] - df["predicted_change"]

forecast_std = df["total_pred"].std()
real_std = df["real_total"].std()

compression_ratio = forecast_std / real_std if real_std != 0 else None

df["weekday"] = df["date"].dt.day_name()

df["is_big_error"] = (df["pct_error"] >= 20).astype(int)

print("=" * 70)
print("VOLATILITY DIAGNOSTICS")
print("=" * 70)

print("\nForecast volatility:", round(forecast_std, 2))
print("Real volatility:", round(real_std, 2))
print("Compression ratio:", round(compression_ratio, 3))

print("\nSe il compression ratio è molto sotto 1, il modello è troppo piatto.")

print("\nGIORNI CON ERRORE > 20%")
print(
    df[df["is_big_error"] == 1][
        [
            "date",
            "weekday",
            "total_pred",
            "real_total",
            "error",
            "pct_error",
            "predicted_change",
            "real_change",
            "compression_gap",
        ]
    ]
)

print("\nMEDIA ERRORE PER GIORNO SETTIMANA")
print(
    df.groupby("weekday")["pct_error"]
    .mean()
    .sort_values(ascending=False)
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSalvato:", OUTPUT_FILE)
