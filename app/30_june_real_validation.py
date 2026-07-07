import pandas as pd

forecast = pd.read_csv("data/processed/forecast_hierarchical_7_days.csv")
forecast["date"] = pd.to_datetime(forecast["date"])

real = pd.DataFrame({
    "date": pd.to_datetime([
        "2026-06-01",
        "2026-06-02",
        "2026-06-03",
        "2026-06-04",
        "2026-06-05",
        "2026-06-06",
        "2026-06-07",
    ]),
    "real_total": [
        1798.27,
        1494.20,
        1363.02,
        1931.19,
        1946.87,
        1820.94,
        2082.20,
    ]
})

df = forecast.merge(real, on="date", how="inner")

df["error"] = df["total_pred"] - df["real_total"]
df["abs_error"] = df["error"].abs()
df["pct_error"] = df["abs_error"] / df["real_total"] * 100

print("\nVALIDAZIONE REALE 1-7 GIUGNO 2026")
print(df[["date", "total_pred", "real_total", "error", "pct_error"]])

print("\nTOTALE PREVISTO:", round(df["total_pred"].sum(), 2))
print("TOTALE REALE:", round(df["real_total"].sum(), 2))
print("ERRORE €:", round(df["total_pred"].sum() - df["real_total"].sum(), 2))
print("ERRORE %:", round((df["total_pred"].sum() - df["real_total"].sum()) / df["real_total"].sum() * 100, 2))

print("\nMAPE GIORNALIERO:", round(df["pct_error"].mean(), 2), "%")

df.to_csv("data/processed/june_real_validation.csv", index=False)
print("\nSalvato: data/processed/june_real_validation.csv")
