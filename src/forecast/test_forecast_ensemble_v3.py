from src.forecast.forecast_ensemble_engine import ForecastEnsembleEngine

engine = ForecastEnsembleEngine()

result = engine.predict_backtest(
    start_date="2026-05-18",
    end_date="2026-05-31",
)

print("=" * 90)
print("FORECAST ENSEMBLE V3 BACKTEST")
print("=" * 90)

if result.empty:
    print("Nessuna riga trovata nel periodo selezionato.")
else:
    print(
        result[
            [
                "date",
                "real_total",
                "ensemble_forecast",
                "error",
                "pct_error",
                "confidence_pct",
                "confidence_label",
                "restaurant_state",
            ]
        ].to_string(index=False)
    )

    print("\nGIORNI:", len(result))
    print("TOTALE PREVISTO:", round(result["ensemble_forecast"].sum(), 2))
    print("TOTALE REALE:", round(result["real_total"].sum(), 2))
    print("ERRORE €:", round(result["ensemble_forecast"].sum() - result["real_total"].sum(), 2))
    print(
        "ERRORE %:",
        round(
            (result["ensemble_forecast"].sum() - result["real_total"].sum())
            / result["real_total"].sum()
            * 100,
            2,
        ),
    )
    print("MAE:", round(result["abs_error"].mean(), 2))
    print("MAPE:", round(result["pct_error"].mean(), 2), "%")

print("\nSalvato: data/processed/forecast_ensemble_v3.csv")
